#include "nrx/inference/image_preprocessor.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>

#include <Windows.h>
#include <directx/d3d12.h>
#include <d3dcompiler.h>
#include <directx/d3dx12.h>
#include <winrt/base.h>

#include "generated/inference/preprocess_cs_embedded.hpp"
#include "nrx/gfx/dx_context.hpp"
#include "nrx/inference/resource_transition.hpp"
#include "nrx/inference/types.hpp"
#include "nrx/utils/dx_helper.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::inference {

namespace {

constexpr Resolution kInputResolution{
    .width = 640,
    .height = 640,
};
constexpr std::uint32_t kInputChannels = 3;
constexpr std::uint32_t kThreadGroupSize = 8;
constexpr std::uint32_t kRootConstantCount = 8;
constexpr DWORD kFenceWaitTimeoutMs = 5000;

// Must match the HLSL cbuffer layout and root constant count.
struct PreprocessConstants {
    std::uint32_t srcWidth;
    std::uint32_t srcHeight;
    std::uint32_t dstWidth;
    std::uint32_t dstHeight;
    float scale;
    float padX;
    float padY;
    float inv255;
};
static_assert((sizeof(PreprocessConstants) / sizeof(std::uint32_t)) == kRootConstantCount);

[[nodiscard]] auto resolveSupportedSrvFormat(DXGI_FORMAT format) -> DXGI_FORMAT {
    switch (format) {
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return format;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return format;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

} // namespace

class ImagePreprocessor::Impl {
  public:
    auto init(nrx::gfx::DxContext* context) -> std::expected<void, InferenceError> {
        reset();

        const auto deviceResult = validateInitContext(context);
        if (!deviceResult) {
            return std::unexpected(deviceResult.error());
        }
        auto* d12Device = *deviceResult;
        dxContext = context;

        if (const auto compileResult = compilePreprocessShader(); !compileResult) {
            reset();
            return std::unexpected(compileResult.error());
        }

        if (const auto pipelineResult = initializeComputePipeline(d12Device); !pipelineResult) {
            reset();
            return std::unexpected(pipelineResult.error());
        }

        if (const auto tensorResult = createPreprocessedTensor(d12Device); !tensorResult) {
            reset();
            return std::unexpected(tensorResult.error());
        }

        const auto uavResult = createOutputUav(d12Device);
        if (!uavResult) {
            reset();
            return std::unexpected(uavResult.error());
        }

        initialized = true;
        return {};
    }

    auto preprocess(ID3D12Resource* inputTexture, const ResourceTransition& transition)
        -> std::expected<PreprocessOutput, InferenceError> {
        if (!initialized || dxContext == nullptr || preprocessedTensor == nullptr) {
            return std::unexpected(InferenceError::NotInitialized);
        }
        if (dxContext->checkDeviceLost()) {
            return std::unexpected(InferenceError::DeviceLost);
        }
        if (inputTexture == nullptr) {
            return std::unexpected(InferenceError::InvalidArguments);
        }

        const auto bindResult = bindInputAndConstants(inputTexture, transition);
        if (!bindResult) {
            return std::unexpected(bindResult.error());
        }

        if (const auto dispatchResult = dispatchAndWait(); !dispatchResult) {
            return std::unexpected(dispatchResult.error());
        }

        return PreprocessOutput{
            .resource = preprocessedTensor.get(),
            .currentState = preprocessedTensorState,
        };
    }

    void reset() {
        initialized = false;
        dxContext = nullptr;
        if (fenceEvent != nullptr) {
            CloseHandle(fenceEvent);
            fenceEvent = nullptr;
        }
        fenceValue = 0;
        commandAllocator = nullptr;
        commandList = nullptr;
        computePipelineState = nullptr;
        rootSignature = nullptr;
        descriptorHeap = nullptr;
        descriptorSize = 0;
        completionFence = nullptr;
        preprocessedTensorState = D3D12_RESOURCE_STATE_COMMON;
        cachedInputTexture = nullptr;
        cachedSrvFormat = DXGI_FORMAT_UNKNOWN;
        shaderBlob = nullptr;
        preprocessedTensor = nullptr;
    }

  private:
    auto validateInitContext(nrx::gfx::DxContext* context)
        -> std::expected<ID3D12Device*, InferenceError> {
        if (context == nullptr) {
            return std::unexpected(InferenceError::InvalidArguments);
        }
        if (context->checkDeviceLost()) {
            return std::unexpected(InferenceError::DeviceLost);
        }

        auto* d12Device = context->getD12Device();
        if (d12Device == nullptr || context->getD12Queue() == nullptr) {
            return std::unexpected(InferenceError::PreprocessFailed);
        }
        return d12Device;
    }

    auto compilePreprocessShader() -> std::expected<void, InferenceError> {
        winrt::com_ptr<ID3DBlob> shaderErrors;
        const HRESULT compileHr =
            D3DCompile(shaders::kPreprocessCsHlslSource, shaders::kPreprocessCsHlslSourceLength,
                       "preprocess_cs.hlsl", nullptr, nullptr, "main", "cs_5_0",
                       D3DCOMPILE_ENABLE_STRICTNESS, 0, shaderBlob.put(), shaderErrors.put());

        if (SUCCEEDED(compileHr)) {
            return {};
        }

        if (shaderErrors) {
            NRX_ERROR("ImagePreprocessor shader compile failed: {}",
                      static_cast<const char*>(shaderErrors->GetBufferPointer()));
        } else {
            NRX_ERROR("ImagePreprocessor shader compile failed: {}",
                      nrx::utils::DxHelper::getErrorString(compileHr));
        }
        return std::unexpected(InferenceError::PreprocessFailed);
    }

    auto initializeComputePipeline(ID3D12Device* d12Device)
        -> std::expected<void, InferenceError> {
        if (const auto signatureResult = createRootSignature(d12Device); !signatureResult) {
            return std::unexpected(signatureResult.error());
        }
        if (const auto pipelineResult = createPipelineState(d12Device); !pipelineResult) {
            return std::unexpected(pipelineResult.error());
        }
        if (const auto queueResult = createQueueResources(d12Device); !queueResult) {
            return std::unexpected(queueResult.error());
        }
        return {};
    }

    auto createPreprocessedTensor(ID3D12Device* d12Device)
        -> std::expected<void, InferenceError> {
        const auto tensorBytes = static_cast<std::uint64_t>(kInputResolution.width) *
                                 kInputResolution.height * kInputChannels * sizeof(float);

        const auto bufferResult = nrx::utils::DxHelper::createBuffer(
            d12Device, tensorBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_HEAP_TYPE_DEFAULT);
        if (!bufferResult) {
            NRX_ERROR("ImagePreprocessor failed to create preprocessed tensor buffer: {}",
                      nrx::utils::DxHelper::getErrorString(bufferResult.error()));
            return std::unexpected(InferenceError::PreprocessFailed);
        }

        preprocessedTensor = bufferResult.value();
        return {};
    }

    auto transitionTrackedResource(ID3D12Resource* resource,
                                   D3D12_RESOURCE_STATES& trackedState,
                                   D3D12_RESOURCE_STATES targetState) -> void {
        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        if (trackedState == targetState) {
            return;
        }

        D3D12_RESOURCE_BARRIER transitionBarrier{};
        transitionBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transitionBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        transitionBarrier.Transition.pResource = resource;
        transitionBarrier.Transition.StateBefore = trackedState;
        transitionBarrier.Transition.StateAfter = targetState;
        transitionBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        commandList->ResourceBarrier(1, &transitionBarrier);
        trackedState = targetState;
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    }

    auto createRootSignature(ID3D12Device* d12Device) -> std::expected<void, InferenceError> {
        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        D3D12_DESCRIPTOR_RANGE1 srvRange{};
        srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors = 1;
        srvRange.BaseShaderRegister = 0;
        srvRange.RegisterSpace = 0;
        srvRange.OffsetInDescriptorsFromTableStart = 0;
        srvRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;

        D3D12_DESCRIPTOR_RANGE1 uavRange{};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = 1;
        uavRange.BaseShaderRegister = 0;
        uavRange.RegisterSpace = 0;
        uavRange.OffsetInDescriptorsFromTableStart = 0;
        uavRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;

        std::array<D3D12_ROOT_PARAMETER1, 3> rootParameters{};
        rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParameters[0].Constants.ShaderRegister =
            0;
        rootParameters[0].Constants.RegisterSpace =
            0;
        rootParameters[0].Constants.Num32BitValues =
            kRootConstantCount;

        rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParameters[1].DescriptorTable.NumDescriptorRanges =
            1;
        rootParameters[1].DescriptorTable.pDescriptorRanges =
            &srvRange;

        rootParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParameters[2].DescriptorTable.NumDescriptorRanges =
            1;
        rootParameters[2].DescriptorTable.pDescriptorRanges =
            &uavRange;

        D3D12_STATIC_SAMPLER_DESC staticSampler{};
        staticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        staticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        staticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        staticSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        staticSampler.MipLODBias = 0.0F;
        staticSampler.MaxAnisotropy = 1;
        staticSampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        staticSampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
        staticSampler.MinLOD = 0.0F;
        staticSampler.MaxLOD = D3D12_FLOAT32_MAX;
        staticSampler.ShaderRegister = 0;
        staticSampler.RegisterSpace = 0;
        staticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDescription{};
        rootSignatureDescription.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        rootSignatureDescription.Desc_1_1.NumParameters = static_cast<UINT>(rootParameters.size());
        rootSignatureDescription.Desc_1_1.pParameters = rootParameters.data();
        rootSignatureDescription.Desc_1_1.NumStaticSamplers = 1;
        rootSignatureDescription.Desc_1_1.pStaticSamplers = &staticSampler;
        rootSignatureDescription.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        winrt::com_ptr<ID3DBlob> serializedRootSignature;
        winrt::com_ptr<ID3DBlob> rootSignatureErrors;
        const HRESULT serializeHr = D3D12SerializeVersionedRootSignature(
            &rootSignatureDescription, serializedRootSignature.put(), rootSignatureErrors.put());
        if (FAILED(serializeHr)) {
            if (rootSignatureErrors != nullptr) {
                NRX_ERROR("ImagePreprocessor root signature serialization failed: {}",
                          static_cast<const char*>(rootSignatureErrors->GetBufferPointer()));
            } else {
                NRX_ERROR("ImagePreprocessor root signature serialization failed: {}",
                          nrx::utils::DxHelper::getErrorString(serializeHr));
            }
            return std::unexpected(InferenceError::PreprocessFailed);
        }

        const HRESULT createRootSignatureHr = d12Device->CreateRootSignature(
            0, serializedRootSignature->GetBufferPointer(),
            serializedRootSignature->GetBufferSize(), IID_PPV_ARGS(rootSignature.put()));
        NRX_DX_CHECK(createRootSignatureHr, "ImagePreprocessor failed to create root signature",
                     InferenceError::PreprocessFailed);
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
        return {};
    }

    auto createPipelineState(ID3D12Device* d12Device) -> std::expected<void, InferenceError> {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDescription{};
        psoDescription.pRootSignature = rootSignature.get();
        psoDescription.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
        psoDescription.CS.BytecodeLength = shaderBlob->GetBufferSize();

        const HRESULT createPsoHr = d12Device->CreateComputePipelineState(
            &psoDescription, IID_PPV_ARGS(computePipelineState.put()));
        NRX_DX_CHECK(createPsoHr, "ImagePreprocessor failed to create compute PSO",
                     InferenceError::PreprocessFailed);
        return {};
    }

    auto createQueueResources(ID3D12Device* d12Device) -> std::expected<void, InferenceError> {
        D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
        heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDescription.NumDescriptors = 2;
        heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        heapDescription.NodeMask = 0;
        const HRESULT heapHr =
            d12Device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(descriptorHeap.put()));
        NRX_DX_CHECK(heapHr, "ImagePreprocessor failed to create descriptor heap",
                     InferenceError::PreprocessFailed);
        descriptorSize =
            d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        const HRESULT allocatorHr = d12Device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(commandAllocator.put()));
        NRX_DX_CHECK(allocatorHr, "ImagePreprocessor failed to create command allocator",
                     InferenceError::PreprocessFailed);

        const HRESULT listHr = d12Device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.get(), computePipelineState.get(),
            IID_PPV_ARGS(commandList.put()));
        NRX_DX_CHECK(listHr, "ImagePreprocessor failed to create command list",
                     InferenceError::PreprocessFailed);

        const HRESULT closeListHr = commandList->Close();
        NRX_DX_CHECK(closeListHr, "ImagePreprocessor failed to close initial command list",
                     InferenceError::PreprocessFailed);

        const HRESULT fenceHr =
            d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(completionFence.put()));
        NRX_DX_CHECK(fenceHr, "ImagePreprocessor failed to create completion fence",
                     InferenceError::PreprocessFailed);

        fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent == nullptr) {
            NRX_ERROR("ImagePreprocessor failed to create fence event");
            return std::unexpected(InferenceError::PreprocessFailed);
        }
        fenceValue = 0;
        return {};
    }

    auto createOutputUav(ID3D12Device* d12Device) -> std::expected<void, InferenceError> {
        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDescription{};
        uavDescription.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDescription.Format = DXGI_FORMAT_UNKNOWN;
        uavDescription.Buffer.FirstElement = 0;
        uavDescription.Buffer.Flags =
            D3D12_BUFFER_UAV_FLAG_NONE;
        uavDescription.Buffer.StructureByteStride =
            sizeof(float);
        uavDescription.Buffer.NumElements =
            kInputResolution.width * kInputResolution.height *
            kInputChannels;

        d12Device->CreateUnorderedAccessView(preprocessedTensor.get(), nullptr, &uavDescription,
                                             descriptorCpuHandle(1));
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
        return {};
    }

    auto bindInputAndConstants(ID3D12Resource* inputTexture,
                               const ResourceTransition& resourceTransition)
        -> std::expected<void, InferenceError> {
        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        auto* d12Device = dxContext->getD12Device();
        if (d12Device == nullptr || descriptorHeap == nullptr || commandList == nullptr ||
            commandAllocator == nullptr || computePipelineState == nullptr ||
            rootSignature == nullptr) {
            return std::unexpected(InferenceError::NotInitialized);
        }

        const D3D12_RESOURCE_DESC inputDescription = inputTexture->GetDesc();
        if (inputDescription.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
            inputDescription.Width == 0 || inputDescription.Height == 0 ||
            inputDescription.Format == DXGI_FORMAT_UNKNOWN) {
            return std::unexpected(InferenceError::InvalidArguments);
        }
        if (inputDescription.SampleDesc.Count > 1) {
            return std::unexpected(InferenceError::InvalidArguments);
        }
        const DXGI_FORMAT srvFormat = resolveSupportedSrvFormat(inputDescription.Format);
        if (srvFormat == DXGI_FORMAT_UNKNOWN) {
            return std::unexpected(InferenceError::PreprocessFailed);
        }

        const bool shouldRefreshSrv =
            (cachedInputTexture != inputTexture) || (cachedSrvFormat != srvFormat);
        if (shouldRefreshSrv) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDescription{};
            srvDescription.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDescription.Format = srvFormat;
            srvDescription.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDescription.Texture2D.MostDetailedMip =
                0;
            srvDescription.Texture2D.MipLevels = 1;
            srvDescription.Texture2D.PlaneSlice = 0;
            srvDescription.Texture2D.ResourceMinLODClamp =
                0.0F;
            d12Device->CreateShaderResourceView(inputTexture, &srvDescription, descriptorCpuHandle(0));
            cachedInputTexture = inputTexture;
            cachedSrvFormat = srvFormat;
        }

        const auto srcWidth = static_cast<float>(inputDescription.Width);
        const auto srcHeight = static_cast<float>(inputDescription.Height);
        const auto dstWidth = static_cast<float>(kInputResolution.width);
        const auto dstHeight = static_cast<float>(kInputResolution.height);
        const float scale = std::min(dstWidth / srcWidth, dstHeight / srcHeight);
        const float scaledWidth = srcWidth * scale;
        const float scaledHeight = srcHeight * scale;

        const PreprocessConstants constants{
            .srcWidth = static_cast<std::uint32_t>(inputDescription.Width),
            .srcHeight = static_cast<std::uint32_t>(inputDescription.Height),
            .dstWidth = kInputResolution.width,
            .dstHeight = kInputResolution.height,
            .scale = scale,
            .padX = (dstWidth - scaledWidth) * 0.5F,
            .padY = (dstHeight - scaledHeight) * 0.5F,
            .inv255 = 1.0F / 255.0F,
        };

        const HRESULT resetAllocatorHr = commandAllocator->Reset();
        NRX_DX_CHECK(resetAllocatorHr, "ImagePreprocessor failed to reset command allocator",
                     InferenceError::PreprocessFailed);

        const HRESULT resetListHr =
            commandList->Reset(commandAllocator.get(), computePipelineState.get());
        NRX_DX_CHECK(resetListHr, "ImagePreprocessor failed to reset command list",
                     InferenceError::PreprocessFailed);

        std::array<ID3D12DescriptorHeap*, 1> descriptorHeaps = {descriptorHeap.get()};
        commandList->SetDescriptorHeaps(static_cast<UINT>(descriptorHeaps.size()),
                                        descriptorHeaps.data());
        commandList->SetComputeRootSignature(rootSignature.get());
        commandList->SetComputeRoot32BitConstants(0, kRootConstantCount, &constants, 0);
        commandList->SetComputeRootDescriptorTable(1, descriptorGpuHandle(0));
        commandList->SetComputeRootDescriptorTable(2, descriptorGpuHandle(1));

        auto inputTrackedState = resourceTransition.fromState;

        transitionTrackedResource(inputTexture, inputTrackedState,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transitionTrackedResource(preprocessedTensor.get(), preprocessedTensorState,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        const UINT groupCountX = (kInputResolution.width + kThreadGroupSize - 1) / kThreadGroupSize;
        const UINT groupCountY =
            (kInputResolution.height + kThreadGroupSize - 1) / kThreadGroupSize;
        commandList->Dispatch(groupCountX, groupCountY, 1);

        D3D12_RESOURCE_BARRIER uavBarrier{};
        uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        uavBarrier.UAV.pResource =
            preprocessedTensor.get();
        commandList->ResourceBarrier(1, &uavBarrier);

        transitionTrackedResource(preprocessedTensor.get(), preprocessedTensorState,
                                  D3D12_RESOURCE_STATE_COMMON);
        transitionTrackedResource(inputTexture, inputTrackedState, resourceTransition.toState);

        const HRESULT closeListHr = commandList->Close();
        NRX_DX_CHECK(closeListHr, "ImagePreprocessor failed to close command list",
                     InferenceError::PreprocessFailed);

        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
        return {};
    }

    auto dispatchAndWait() -> std::expected<void, InferenceError> {
        auto* d12Queue = dxContext->getD12Queue();
        if (d12Queue == nullptr || commandList == nullptr || completionFence == nullptr ||
            fenceEvent == nullptr) {
            return std::unexpected(InferenceError::NotInitialized);
        }

        const std::array<ID3D12CommandList*, 1> commandLists = {commandList.get()};
        d12Queue->ExecuteCommandLists(static_cast<UINT>(commandLists.size()), commandLists.data());

        fenceValue += 1;
        const HRESULT signalHr = d12Queue->Signal(completionFence.get(), fenceValue);
        NRX_DX_CHECK(signalHr, "ImagePreprocessor failed to signal completion fence",
                     InferenceError::PreprocessFailed);

        if (completionFence->GetCompletedValue() < fenceValue) {
            const HRESULT eventHr = completionFence->SetEventOnCompletion(fenceValue, fenceEvent);
            NRX_DX_CHECK(eventHr, "ImagePreprocessor failed to set fence completion event",
                         InferenceError::PreprocessFailed);
            const DWORD waitResult = WaitForSingleObject(fenceEvent, kFenceWaitTimeoutMs);
            if (waitResult == WAIT_OBJECT_0) {
                return {};
            }
            if (waitResult == WAIT_TIMEOUT) {
                NRX_ERROR("ImagePreprocessor fence wait timed out after {} ms", kFenceWaitTimeoutMs);
            } else {
                const auto waitError =
                    HRESULT_FROM_WIN32(static_cast<unsigned int>(GetLastError()));
                NRX_ERROR("ImagePreprocessor fence wait failed: {}",
                          nrx::utils::DxHelper::getErrorString(static_cast<std::int32_t>(waitError)));
            }
            if (dxContext != nullptr) {
                dxContext->notifyDeviceLost();
            }
            return std::unexpected(InferenceError::DeviceLost);
        }

        return {};
    }

    [[nodiscard]] auto descriptorCpuHandle(UINT index) const -> D3D12_CPU_DESCRIPTOR_HANDLE {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(),
                                             static_cast<INT>(index), static_cast<INT>(descriptorSize));
    }

    [[nodiscard]] auto descriptorGpuHandle(UINT index) const -> D3D12_GPU_DESCRIPTOR_HANDLE {
        return CD3DX12_GPU_DESCRIPTOR_HANDLE(descriptorHeap->GetGPUDescriptorHandleForHeapStart(),
                                             static_cast<INT>(index), static_cast<INT>(descriptorSize));
    }

    bool initialized{false};
    nrx::gfx::DxContext* dxContext{nullptr};
    winrt::com_ptr<ID3D12RootSignature> rootSignature;
    winrt::com_ptr<ID3D12PipelineState> computePipelineState;
    winrt::com_ptr<ID3D12DescriptorHeap> descriptorHeap;
    UINT descriptorSize{0};
    winrt::com_ptr<ID3D12CommandAllocator> commandAllocator;
    winrt::com_ptr<ID3D12GraphicsCommandList> commandList;
    winrt::com_ptr<ID3D12Fence> completionFence;
    HANDLE fenceEvent{nullptr};
    std::uint64_t fenceValue{0};
    D3D12_RESOURCE_STATES preprocessedTensorState{D3D12_RESOURCE_STATE_COMMON};
    ID3D12Resource* cachedInputTexture{nullptr};
    DXGI_FORMAT cachedSrvFormat{DXGI_FORMAT_UNKNOWN};
    winrt::com_ptr<ID3DBlob> shaderBlob;
    winrt::com_ptr<ID3D12Resource> preprocessedTensor;
};

ImagePreprocessor::ImagePreprocessor() : impl(std::make_unique<Impl>()) {}

ImagePreprocessor::~ImagePreprocessor() = default;

auto ImagePreprocessor::init(nrx::gfx::DxContext* dxContext)
    -> std::expected<void, InferenceError> {
    return impl->init(dxContext);
}

auto ImagePreprocessor::preprocess(ID3D12Resource* inputTexture, const ResourceTransition& transition)
    -> std::expected<PreprocessOutput, InferenceError> {
    return impl->preprocess(inputTexture, transition);
}

void ImagePreprocessor::reset() { impl->reset(); }

} // namespace nrx::inference
