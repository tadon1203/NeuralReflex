#include "inference/postprocessor_internal.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>

#include <d3dcompiler.h>
#include <directx/d3d12.h>
#include <winrt/base.h>

#include "generated/inference/post_compact_cs_embedded.hpp"
#include "generated/inference/post_decode_filter_cs_embedded.hpp"
#include "generated/inference/post_nms_cs_embedded.hpp"
#include "nrx/gfx/dx_context.hpp"
#include "nrx/utils/dx_helper.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::inference {

namespace {

constexpr UINT kPostSrvCount = 5;
constexpr UINT kPostUavCount = 7;
constexpr UINT kPostRootConstantCount = 12;

[[nodiscard]] auto buildRootSignature(ID3D12Device* d12Device)
    -> std::expected<winrt::com_ptr<ID3D12RootSignature>, InferenceError> {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    D3D12_DESCRIPTOR_RANGE1 srvRange{};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = kPostSrvCount;
    srvRange.BaseShaderRegister = 0;
    srvRange.RegisterSpace = 0;
    srvRange.OffsetInDescriptorsFromTableStart = 0;
    srvRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;

    D3D12_DESCRIPTOR_RANGE1 uavRange{};
    uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange.NumDescriptors = kPostUavCount;
    uavRange.BaseShaderRegister = 0;
    uavRange.RegisterSpace = 0;
    uavRange.OffsetInDescriptorsFromTableStart = 0;
    uavRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;

    std::array<D3D12_ROOT_PARAMETER1, 3> rootParameters{};
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[0].Constants.ShaderRegister = 0;
    rootParameters[0].Constants.RegisterSpace = 0;
    rootParameters[0].Constants.Num32BitValues = kPostRootConstantCount;

    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[1].DescriptorTable.pDescriptorRanges = &srvRange;

    rootParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[2].DescriptorTable.pDescriptorRanges = &uavRange;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDescription{};
    rootSignatureDescription.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rootSignatureDescription.Desc_1_1.NumParameters = static_cast<UINT>(rootParameters.size());
    rootSignatureDescription.Desc_1_1.pParameters = rootParameters.data();
    rootSignatureDescription.Desc_1_1.NumStaticSamplers = 0;
    rootSignatureDescription.Desc_1_1.pStaticSamplers = nullptr;
    rootSignatureDescription.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    winrt::com_ptr<ID3DBlob> serializedRootSignature;
    winrt::com_ptr<ID3DBlob> rootSignatureErrors;
    const HRESULT serializeHr = D3D12SerializeVersionedRootSignature(
        &rootSignatureDescription, serializedRootSignature.put(), rootSignatureErrors.put());
    if (FAILED(serializeHr)) {
        if (rootSignatureErrors != nullptr) {
            NRX_ERROR("Postprocessor root signature serialization failed: {}",
                      static_cast<const char*>(rootSignatureErrors->GetBufferPointer()));
        } else {
            NRX_ERROR("Postprocessor root signature serialization failed: {}",
                      nrx::utils::DxHelper::getErrorString(serializeHr));
        }
        return std::unexpected(InferenceError::PostprocessFailed);
    }

    winrt::com_ptr<ID3D12RootSignature> rootSignature;
    const HRESULT createRootSignatureHr = d12Device->CreateRootSignature(
        0, serializedRootSignature->GetBufferPointer(), serializedRootSignature->GetBufferSize(),
        IID_PPV_ARGS(rootSignature.put()));
    NRX_DX_CHECK(createRootSignatureHr, "Postprocessor failed to create root signature",
                 InferenceError::PostprocessFailed);
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)

    return rootSignature;
}

[[nodiscard]] auto compileShader(std::string_view source, std::size_t sourceLength,
                                 const char* sourceName)
    -> std::expected<winrt::com_ptr<ID3DBlob>, InferenceError> {
    winrt::com_ptr<ID3DBlob> shaderBlob;
    winrt::com_ptr<ID3DBlob> shaderErrors;
    const HRESULT compileHr =
        D3DCompile(source.data(), sourceLength, sourceName, nullptr, nullptr, "main", "cs_5_0",
                   D3DCOMPILE_ENABLE_STRICTNESS, 0, shaderBlob.put(), shaderErrors.put());
    if (FAILED(compileHr)) {
        if (shaderErrors != nullptr) {
            NRX_ERROR("Postprocessor shader compile failed ({}): {}", sourceName,
                      static_cast<const char*>(shaderErrors->GetBufferPointer()));
        } else {
            NRX_ERROR("Postprocessor shader compile failed ({}): {}", sourceName,
                      nrx::utils::DxHelper::getErrorString(compileHr));
        }
        return std::unexpected(InferenceError::PostprocessFailed);
    }

    return shaderBlob;
}

[[nodiscard]] auto createComputePipelineState(ID3D12Device* d12Device,
                                              ID3D12RootSignature* rootSignature,
                                              ID3DBlob* shaderBlob)
    -> std::expected<winrt::com_ptr<ID3D12PipelineState>, InferenceError> {
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDescription{};
    psoDescription.pRootSignature = rootSignature;
    psoDescription.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
    psoDescription.CS.BytecodeLength = shaderBlob->GetBufferSize();

    winrt::com_ptr<ID3D12PipelineState> pipelineState;
    const HRESULT createPsoHr =
        d12Device->CreateComputePipelineState(&psoDescription, IID_PPV_ARGS(pipelineState.put()));
    NRX_DX_CHECK(createPsoHr, "Postprocessor failed to create compute PSO",
                 InferenceError::PostprocessFailed);
    return pipelineState;
}

} // namespace

Postprocessor::Impl::Impl(Config configValue) : config(configValue) {}

auto Postprocessor::Impl::init(nrx::gfx::DxContext* context, std::span<const int64_t> outputShape,
                               Resolution inputResolutionValue)
    -> std::expected<void, InferenceError> {
    reset();

    const auto validateResult = validateInitInputs(context, outputShape);
    if (!validateResult) {
        return std::unexpected(validateResult.error());
    }
    auto* d12Device = validateResult.value();

    dxContext = context;
    inputResolution = inputResolutionValue;

    if (const auto shapeResult = configureOutputShape(outputShape); !shapeResult) {
        return std::unexpected(shapeResult.error());
    }
    if (const auto pipelineResult = initializePipelineResources(d12Device); !pipelineResult) {
        return std::unexpected(pipelineResult.error());
    }
    if (const auto descriptorResult = initializeDescriptorHeap(d12Device); !descriptorResult) {
        return std::unexpected(descriptorResult.error());
    }
    if (const auto bufferResult = initializeBuffersAndDescriptors(d12Device); !bufferResult) {
        return std::unexpected(bufferResult.error());
    }
    if (const auto commandResult = initializeCommandAndSyncObjects(d12Device); !commandResult) {
        return std::unexpected(commandResult.error());
    }

    initialized = true;
    return {};
}

auto Postprocessor::Impl::validateInitInputs(nrx::gfx::DxContext* context,
                                             std::span<const int64_t> outputShape) const
    -> std::expected<ID3D12Device*, InferenceError> {
    if (context == nullptr || outputShape.size() < 3) {
        return std::unexpected(InferenceError::InvalidArguments);
    }
    if (context->checkDeviceLost()) {
        return std::unexpected(InferenceError::DeviceLost);
    }
    if (config.classCount == 0 || config.maxDetections == 0) {
        return std::unexpected(InferenceError::InvalidArguments);
    }

    auto* d12Device = context->getD12Device();
    if (d12Device == nullptr || context->getD12Queue() == nullptr) {
        return std::unexpected(InferenceError::PostprocessFailed);
    }

    return d12Device;
}

auto Postprocessor::Impl::configureOutputShape(std::span<const int64_t> outputShape)
    -> std::expected<void, InferenceError> {
    const auto dim1 = outputShape[1];
    const auto dim2 = outputShape[2];
    if (dim1 <= 0 || dim2 <= 0) {
        return std::unexpected(InferenceError::PostprocessFailed);
    }

    const auto anchorMajorCandidate =
        dim2 >= static_cast<int64_t>(kMinDetAttributes) && dim1 > dim2;
    if (anchorMajorCandidate) {
        anchorCount = static_cast<std::uint32_t>(dim1);
        attributeCount = static_cast<std::uint32_t>(dim2);
        layout = DetectionLayout::AnchorMajor;
    } else {
        attributeCount = static_cast<std::uint32_t>(dim1);
        anchorCount = static_cast<std::uint32_t>(dim2);
        layout = DetectionLayout::AttributeMajor;
    }
    if (attributeCount < kMinDetAttributes || anchorCount == 0) {
        return std::unexpected(InferenceError::PostprocessFailed);
    }
    outputElementCount = static_cast<std::uint64_t>(attributeCount) * anchorCount;

    if (attributeCount == (4 + config.classCount)) {
        useObjectness = false;
        classStartIndex = 4;
        activeClassCount = config.classCount;
        return {};
    }
    if (attributeCount >= (5 + config.classCount)) {
        useObjectness = true;
        classStartIndex = 5;
        activeClassCount = config.classCount;
        return {};
    }
    if (attributeCount == 5) {
        useObjectness = false;
        classStartIndex = 4;
        activeClassCount = 1;
        return {};
    }

    return std::unexpected(InferenceError::PostprocessFailed);
}

auto Postprocessor::Impl::initializePipelineResources(ID3D12Device* d12Device)
    -> std::expected<void, InferenceError> {
    const auto rootSignatureResult = buildRootSignature(d12Device);
    if (!rootSignatureResult) {
        return std::unexpected(rootSignatureResult.error());
    }
    rootSignature = rootSignatureResult.value();

    const auto decodeShaderResult = compileShader(shaders::kPostDecodeFilterCsHlslSource,
                                                  shaders::kPostDecodeFilterCsHlslSourceLength,
                                                  "post_decode_filter_cs.hlsl");
    if (!decodeShaderResult) {
        return std::unexpected(decodeShaderResult.error());
    }
    const auto nmsShaderResult =
        compileShader(shaders::kPostNmsCsHlslSource, shaders::kPostNmsCsHlslSourceLength,
                      "post_nms_cs.hlsl");
    if (!nmsShaderResult) {
        return std::unexpected(nmsShaderResult.error());
    }
    const auto compactShaderResult = compileShader(shaders::kPostCompactCsHlslSource,
                                                   shaders::kPostCompactCsHlslSourceLength,
                                                   "post_compact_cs.hlsl");
    if (!compactShaderResult) {
        return std::unexpected(compactShaderResult.error());
    }

    const auto decodePsoResult =
        createComputePipelineState(d12Device, rootSignature.get(), decodeShaderResult.value().get());
    if (!decodePsoResult) {
        return std::unexpected(decodePsoResult.error());
    }
    decodePipelineState = decodePsoResult.value();

    const auto nmsPsoResult =
        createComputePipelineState(d12Device, rootSignature.get(), nmsShaderResult.value().get());
    if (!nmsPsoResult) {
        return std::unexpected(nmsPsoResult.error());
    }
    nmsPipelineState = nmsPsoResult.value();

    const auto compactPsoResult =
        createComputePipelineState(d12Device, rootSignature.get(), compactShaderResult.value().get());
    if (!compactPsoResult) {
        return std::unexpected(compactPsoResult.error());
    }
    compactPipelineState = compactPsoResult.value();

    return {};
}

auto Postprocessor::Impl::initializeDescriptorHeap(ID3D12Device* d12Device)
    -> std::expected<void, InferenceError> {
    D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
    heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDescription.NumDescriptors = DescriptorTableLayout::kDescriptorCount;
    heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    const HRESULT createHeapHr =
        d12Device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(descriptorHeap.put()));
    NRX_DX_CHECK(createHeapHr, "Postprocessor descriptor heap creation failed",
                 InferenceError::PostprocessFailed);
    descriptorSize = d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return {};
}

auto Postprocessor::Impl::initializeBuffersAndDescriptors(ID3D12Device* d12Device)
    -> std::expected<void, InferenceError> {
    const auto candidateElementCount = static_cast<std::uint64_t>(anchorCount);
    const auto maxDetectionCount = static_cast<std::uint64_t>(config.maxDetections);

    const std::array bufferSpecs{
        BufferSpec{.resource = &candidateBox,
                   .sizeBytes = sizeof(float) * 4 * candidateElementCount,
                   .kind = BufferKind::Uav},
        BufferSpec{.resource = &candidateScoreClass,
                   .sizeBytes = sizeof(float) * 2 * candidateElementCount,
                   .kind = BufferKind::Uav},
        BufferSpec{.resource = &candidateCount,
                   .sizeBytes = sizeof(std::uint32_t),
                   .kind = BufferKind::Uav},
        BufferSpec{.resource = &suppressed,
                   .sizeBytes = sizeof(std::uint32_t) * candidateElementCount,
                   .kind = BufferKind::Uav},
        BufferSpec{.resource = &finalBox,
                   .sizeBytes = sizeof(float) * 4 * maxDetectionCount,
                   .kind = BufferKind::Uav},
        BufferSpec{.resource = &finalScoreClass,
                   .sizeBytes = sizeof(float) * 2 * maxDetectionCount,
                   .kind = BufferKind::Uav},
        BufferSpec{.resource = &finalCount,
                   .sizeBytes = sizeof(std::uint32_t),
                   .kind = BufferKind::Uav},
        BufferSpec{.resource = &readbackFinalBox,
                   .sizeBytes = sizeof(float) * 4 * maxDetectionCount,
                   .kind = BufferKind::Readback},
        BufferSpec{.resource = &readbackFinalScoreClass,
                   .sizeBytes = sizeof(float) * 2 * maxDetectionCount,
                   .kind = BufferKind::Readback},
        BufferSpec{.resource = &readbackFinalCount,
                   .sizeBytes = sizeof(std::uint32_t),
                   .kind = BufferKind::Readback},
    };

    for (const auto& spec : bufferSpecs) {
        const auto bufferResult = (spec.kind == BufferKind::Uav)
                                      ? nrx::utils::DxHelper::createUavBuffer(d12Device, spec.sizeBytes)
                                      : nrx::utils::DxHelper::createReadbackBuffer(d12Device,
                                                                                   spec.sizeBytes);
        if (!bufferResult) {
            NRX_ERROR("Postprocessor buffer creation failed: {}",
                      nrx::utils::DxHelper::getErrorString(bufferResult.error()));
            reset();
            return std::unexpected(InferenceError::PostprocessFailed);
        }
        *spec.resource = bufferResult.value();
    }

    createStaticDescriptors(d12Device);
    return {};
}

auto Postprocessor::Impl::initializeCommandAndSyncObjects(ID3D12Device* d12Device)
    -> std::expected<void, InferenceError> {
    const HRESULT createAllocatorHr = d12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(commandAllocator.put()));
    NRX_DX_CHECK(createAllocatorHr, "Postprocessor command allocator creation failed",
                 InferenceError::PostprocessFailed);

    const HRESULT createListHr = d12Device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.get(), decodePipelineState.get(),
        IID_PPV_ARGS(commandList.put()));
    NRX_DX_CHECK(createListHr, "Postprocessor command list creation failed",
                 InferenceError::PostprocessFailed);
    const HRESULT closeListHr = commandList->Close();
    NRX_DX_CHECK(closeListHr, "Postprocessor initial command list close failed",
                 InferenceError::PostprocessFailed);

    const HRESULT createFenceHr =
        d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(completionFence.put()));
    NRX_DX_CHECK(createFenceHr, "Postprocessor fence creation failed", InferenceError::PostprocessFailed);
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (fenceEvent == nullptr) {
        return std::unexpected(InferenceError::PostprocessFailed);
    }

    return {};
}

auto Postprocessor::Impl::createStructuredSrv(ID3D12Device* d12Device, ID3D12Resource* resource,
                                              const StructuredViewSpec& viewSpec) -> void {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    D3D12_SHADER_RESOURCE_VIEW_DESC description{};
    description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    description.Buffer.FirstElement = 0;
    description.Buffer.NumElements = viewSpec.numElements;
    description.Buffer.StructureByteStride = viewSpec.stride;
    description.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    d12Device->CreateShaderResourceView(resource, &description, descriptorCpuHandle(viewSpec.descriptorIndex));
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
}

auto Postprocessor::Impl::createStructuredUav(ID3D12Device* d12Device, ID3D12Resource* resource,
                                              const StructuredViewSpec& viewSpec) -> void {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    description.Buffer.FirstElement = 0;
    description.Buffer.NumElements = viewSpec.numElements;
    description.Buffer.StructureByteStride = viewSpec.stride;
    description.Buffer.CounterOffsetInBytes = 0;
    description.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    d12Device->CreateUnorderedAccessView(resource, nullptr, &description,
                                         descriptorCpuHandle(viewSpec.descriptorIndex));
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
}

auto Postprocessor::Impl::createStaticDescriptors(ID3D12Device* d12Device) -> void {
    createStructuredSrv(d12Device, candidateBox.get(),
                        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kSrvCandidateBox,
                                           .stride = sizeof(float) * 4,
                                           .numElements = anchorCount});
    createStructuredSrv(
        d12Device, candidateScoreClass.get(),
        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kSrvCandidateScoreClass,
                           .stride = sizeof(float) * 2,
                           .numElements = anchorCount});
    createStructuredSrv(d12Device, candidateCount.get(),
                        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kSrvCandidateCount,
                                           .stride = sizeof(std::uint32_t),
                                           .numElements = 1});
    createStructuredSrv(d12Device, suppressed.get(),
                        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kSrvSuppressed,
                                           .stride = sizeof(std::uint32_t),
                                           .numElements = anchorCount});

    createStructuredUav(d12Device, candidateBox.get(),
                        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kUavCandidateBox,
                                           .stride = sizeof(float) * 4,
                                           .numElements = anchorCount});
    createStructuredUav(
        d12Device, candidateScoreClass.get(),
        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kUavCandidateScoreClass,
                           .stride = sizeof(float) * 2,
                           .numElements = anchorCount});
    createStructuredUav(d12Device, candidateCount.get(),
                        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kUavCandidateCount,
                                           .stride = sizeof(std::uint32_t),
                                           .numElements = 1});
    createStructuredUav(d12Device, suppressed.get(),
                        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kUavSuppressed,
                                           .stride = sizeof(std::uint32_t),
                                           .numElements = anchorCount});
    createStructuredUav(d12Device, finalBox.get(),
                        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kUavFinalBox,
                                           .stride = sizeof(float) * 4,
                                           .numElements = config.maxDetections});
    createStructuredUav(
        d12Device, finalScoreClass.get(),
        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kUavFinalScoreClass,
                           .stride = sizeof(float) * 2,
                           .numElements = config.maxDetections});
    createStructuredUav(d12Device, finalCount.get(),
                        StructuredViewSpec{.descriptorIndex = DescriptorTableLayout::kUavFinalCount,
                                           .stride = sizeof(std::uint32_t),
                                           .numElements = 1});
}

} // namespace nrx::inference
