#include "nrx/inference/postprocessor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <Windows.h>
#include <d3dcompiler.h>
#include <directx/d3d12.h>
#include <directx/d3dx12.h>
#include <winrt/base.h>

#include "generated/inference/post_compact_cs_embedded.hpp"
#include "generated/inference/post_decode_filter_cs_embedded.hpp"
#include "generated/inference/post_nms_cs_embedded.hpp"
#include "nrx/gfx/dx_context.hpp"
#include "nrx/utils/dx_helper.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::inference {

namespace {

constexpr std::size_t kMinDetAttributes = 5;
constexpr UINT kThreadGroupSize = 256;
constexpr UINT kRootConstantCount = 12;

struct DescriptorTableLayout {
    static constexpr UINT kSrvRawOutput = 0;
    static constexpr UINT kSrvCandidateBox = 1;
    static constexpr UINT kSrvCandidateScoreClass = 2;
    static constexpr UINT kSrvCandidateCount = 3;
    static constexpr UINT kSrvSuppressed = 4;

    static constexpr UINT kUavCandidateBox = 5;
    static constexpr UINT kUavCandidateScoreClass = 6;
    static constexpr UINT kUavCandidateCount = 7;
    static constexpr UINT kUavSuppressed = 8;
    static constexpr UINT kUavFinalBox = 9;
    static constexpr UINT kUavFinalScoreClass = 10;
    static constexpr UINT kUavFinalCount = 11;

    static constexpr UINT kSrvCount = 5;
    static constexpr UINT kUavCount = 7;
    static constexpr UINT kDescriptorCount = 12;
};

enum class DetectionLayout : std::uint8_t {
    AttributeMajor = 0,
    AnchorMajor = 1,
};

struct PostConstants {
    std::uint32_t anchorCount;
    std::uint32_t attributeCount;
    std::uint32_t layoutFlag;
    std::uint32_t classCount;
    std::uint32_t useObjectness;
    std::uint32_t classStartIndex;
    std::uint32_t maxDetections;
    std::uint32_t reserved0;
    float inputWidth;
    float inputHeight;
    float scoreThreshold;
    float nmsIouThreshold;
};

struct StructuredViewSpec {
    UINT descriptorIndex;
    UINT stride;
    UINT numElements;
};

[[nodiscard]] auto buildRootSignature(ID3D12Device* d12Device)
    -> std::expected<winrt::com_ptr<ID3D12RootSignature>, InferenceError> {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    D3D12_DESCRIPTOR_RANGE1 srvRange{};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = DescriptorTableLayout::kSrvCount;
    srvRange.BaseShaderRegister = 0;
    srvRange.RegisterSpace = 0;
    srvRange.OffsetInDescriptorsFromTableStart = 0;
    srvRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;

    D3D12_DESCRIPTOR_RANGE1 uavRange{};
    uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange.NumDescriptors = DescriptorTableLayout::kUavCount;
    uavRange.BaseShaderRegister = 0;
    uavRange.RegisterSpace = 0;
    uavRange.OffsetInDescriptorsFromTableStart = 0;
    uavRange.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;

    std::array<D3D12_ROOT_PARAMETER1, 3> rootParameters{};
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[0].Constants.ShaderRegister = 0;
    rootParameters[0].Constants.RegisterSpace = 0;
    rootParameters[0].Constants.Num32BitValues = kRootConstantCount;

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

class Postprocessor::Impl {
  public:
    explicit Impl(Config configValue) : config(configValue) {}

    auto init(nrx::gfx::DxContext* context, std::span<const int64_t> outputShape,
              Resolution inputResolutionValue) -> std::expected<void, InferenceError> {
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

    auto dispatch(ID3D12Resource* rawOutputResource, D3D12_RESOURCE_STATES currentState)
        -> std::expected<void, InferenceError> {
        if (!initialized || dxContext == nullptr || rawOutputResource == nullptr) {
            return std::unexpected(InferenceError::NotInitialized);
        }
        if (dxContext->checkDeviceLost()) {
            return std::unexpected(InferenceError::DeviceLost);
        }

        auto* d12Device = dxContext->getD12Device();
        auto* d12Queue = dxContext->getD12Queue();
        if (d12Device == nullptr || d12Queue == nullptr || commandAllocator == nullptr ||
            commandList == nullptr) {
            return std::unexpected(InferenceError::PostprocessFailed);
        }

        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        D3D12_SHADER_RESOURCE_VIEW_DESC rawOutputSrv{};
        rawOutputSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        rawOutputSrv.Format = DXGI_FORMAT_UNKNOWN;
        rawOutputSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        rawOutputSrv.Buffer.FirstElement = 0;
        rawOutputSrv.Buffer.NumElements = static_cast<UINT>(outputElementCount);
        rawOutputSrv.Buffer.StructureByteStride = sizeof(float);
        rawOutputSrv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        d12Device->CreateShaderResourceView(
            rawOutputResource, &rawOutputSrv,
            descriptorCpuHandle(DescriptorTableLayout::kSrvRawOutput));
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)

        const HRESULT resetAllocatorHr = commandAllocator->Reset();
        NRX_DX_CHECK(resetAllocatorHr, "Postprocessor command allocator reset failed",
                     InferenceError::PostprocessFailed);
        const HRESULT resetListHr =
            commandList->Reset(commandAllocator.get(), decodePipelineState.get());
        NRX_DX_CHECK(resetListHr, "Postprocessor command list reset failed",
                     InferenceError::PostprocessFailed);

        const std::array<ID3D12DescriptorHeap*, 1> descriptorHeaps = {descriptorHeap.get()};
        commandList->SetDescriptorHeaps(static_cast<UINT>(descriptorHeaps.size()),
                                        descriptorHeaps.data());
        commandList->SetComputeRootSignature(rootSignature.get());

        const PostConstants constants{
            .anchorCount = anchorCount,
            .attributeCount = attributeCount,
            .layoutFlag = static_cast<std::uint32_t>(layout),
            .classCount = activeClassCount,
            .useObjectness = useObjectness ? 1U : 0U,
            .classStartIndex = classStartIndex,
            .maxDetections = config.maxDetections,
            .reserved0 = 0,
            .inputWidth = static_cast<float>(inputResolution.width),
            .inputHeight = static_cast<float>(inputResolution.height),
            .scoreThreshold = config.scoreThreshold,
            .nmsIouThreshold = config.nmsIouThreshold,
        };
        commandList->SetComputeRoot32BitConstants(0, kRootConstantCount, &constants, 0);
        commandList->SetComputeRootDescriptorTable(
            1, descriptorGpuHandle(DescriptorTableLayout::kSrvRawOutput));
        commandList->SetComputeRootDescriptorTable(
            2, descriptorGpuHandle(DescriptorTableLayout::kUavCandidateBox));

        auto rawOutputState = currentState;
        transitionResources(
            {
                TransitionRequest{rawOutputResource, &rawOutputState},
                TransitionRequest{candidateBox.get(), &candidateBoxState},
                TransitionRequest{candidateScoreClass.get(), &candidateScoreClassState},
                TransitionRequest{candidateCount.get(), &candidateCountState},
                TransitionRequest{suppressed.get(), &suppressedState},
                TransitionRequest{finalBox.get(), &finalBoxState},
                TransitionRequest{finalScoreClass.get(), &finalScoreClassState},
                TransitionRequest{finalCount.get(), &finalCountState},
            },
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        const auto clearCandidateCountResult =
            clearCounterUav(candidateCount.get(), DescriptorTableLayout::kUavCandidateCount);
        if (!clearCandidateCountResult) {
            return std::unexpected(clearCandidateCountResult.error());
        }

        const UINT groupCount = (anchorCount + kThreadGroupSize - 1) / kThreadGroupSize;
        runDecodeStage(groupCount);
        runNmsStage(groupCount);

        commandList->SetPipelineState(compactPipelineState.get());

        const auto clearFinalCountResult =
            clearCounterUav(finalCount.get(), DescriptorTableLayout::kUavFinalCount);
        if (!clearFinalCountResult) {
            return std::unexpected(clearFinalCountResult.error());
        }

        runCompactStage(groupCount);

        transitionResources(
            {
                TransitionRequest{finalBox.get(), &finalBoxState},
                TransitionRequest{finalScoreClass.get(), &finalScoreClassState},
                TransitionRequest{finalCount.get(), &finalCountState},
            },
            D3D12_RESOURCE_STATE_COPY_SOURCE);

        copyReadbackResources();

        transitionResources(
            {
                TransitionRequest{finalBox.get(), &finalBoxState},
                TransitionRequest{finalScoreClass.get(), &finalScoreClassState},
                TransitionRequest{finalCount.get(), &finalCountState},
                TransitionRequest{rawOutputResource, &rawOutputState},
            },
            D3D12_RESOURCE_STATE_COMMON);

        const HRESULT closeListHr = commandList->Close();
        NRX_DX_CHECK(closeListHr, "Postprocessor command list close failed",
                     InferenceError::PostprocessFailed);

        const std::array<ID3D12CommandList*, 1> commandLists = {commandList.get()};
        d12Queue->ExecuteCommandLists(static_cast<UINT>(commandLists.size()), commandLists.data());

        fenceValue += 1;
        const HRESULT signalHr = d12Queue->Signal(completionFence.get(), fenceValue);
        NRX_DX_CHECK(signalHr, "Postprocessor signal failed", InferenceError::PostprocessFailed);
        if (completionFence->GetCompletedValue() < fenceValue) {
            const HRESULT waitEventHr =
                completionFence->SetEventOnCompletion(fenceValue, fenceEvent);
            NRX_DX_CHECK(waitEventHr, "Postprocessor wait event setup failed",
                         InferenceError::PostprocessFailed);
            WaitForSingleObject(fenceEvent, INFINITE);
        }

        return {};
    }

    auto runDecodeStage(UINT groupCount) -> void {
        commandList->Dispatch(groupCount, 1, 1);
        addUavBarrier(candidateBox.get());
        addUavBarrier(candidateScoreClass.get());
        addUavBarrier(candidateCount.get());
    }

    auto runNmsStage(UINT groupCount) -> void {
        commandList->SetPipelineState(nmsPipelineState.get());
        commandList->Dispatch(groupCount, 1, 1);
        addUavBarrier(suppressed.get());
    }

    auto runCompactStage(UINT groupCount) -> void {
        commandList->Dispatch(groupCount, 1, 1);
        addUavBarrier(finalBox.get());
        addUavBarrier(finalScoreClass.get());
        addUavBarrier(finalCount.get());
    }

    auto copyReadbackResources() -> void {
        commandList->CopyResource(readbackFinalBox.get(), finalBox.get());
        commandList->CopyResource(readbackFinalScoreClass.get(), finalScoreClass.get());
        commandList->CopyResource(readbackFinalCount.get(), finalCount.get());
    }

    auto readbackFinalResults() -> std::expected<DetectionResults, InferenceError> {
        if (!initialized || readbackFinalCount == nullptr || readbackFinalBox == nullptr ||
            readbackFinalScoreClass == nullptr) {
            return std::unexpected(InferenceError::NotInitialized);
        }

        void* mappedCount = nullptr;
        D3D12_RANGE readRangeCount{0, sizeof(std::uint32_t)};
        const HRESULT mapCountHr = readbackFinalCount->Map(0, &readRangeCount, &mappedCount);
        NRX_DX_CHECK(mapCountHr, "Postprocessor count readback map failed",
                     InferenceError::PostprocessFailed);
        const auto finalCountValue =
            std::min(*static_cast<const std::uint32_t*>(mappedCount), config.maxDetections);
        readbackFinalCount->Unmap(0, nullptr);

        if (finalCountValue == 0) {
            return DetectionResults{};
        }

        void* mappedBox = nullptr;
        void* mappedScoreClass = nullptr;
        const auto boxReadSize = static_cast<std::size_t>(finalCountValue) * sizeof(float) * 4;
        const auto scoreClassReadSize =
            static_cast<std::size_t>(finalCountValue) * sizeof(float) * 2;
        D3D12_RANGE boxReadRange{0, boxReadSize};
        D3D12_RANGE scoreClassReadRange{0, scoreClassReadSize};

        const HRESULT mapBoxHr = readbackFinalBox->Map(0, &boxReadRange, &mappedBox);
        NRX_DX_CHECK(mapBoxHr, "Postprocessor final box readback map failed",
                     InferenceError::PostprocessFailed);
        const HRESULT mapScoreClassHr =
            readbackFinalScoreClass->Map(0, &scoreClassReadRange, &mappedScoreClass);
        NRX_DX_CHECK(mapScoreClassHr, "Postprocessor final score/class readback map failed",
                     InferenceError::PostprocessFailed);

        const auto* boxValues = static_cast<const float*>(mappedBox);
        const auto* scoreClassValues = static_cast<const float*>(mappedScoreClass);

        DetectionResults results;
        results.reserve(finalCountValue);
        for (std::uint32_t i = 0; i < finalCountValue; ++i) {
            const auto boxBase = static_cast<std::size_t>(i) * 4;
            const auto scoreClassBase = static_cast<std::size_t>(i) * 2;
            results.push_back(DetectionResult{
                .x = boxValues[boxBase + 0],
                .y = boxValues[boxBase + 1],
                .w = boxValues[boxBase + 2],
                .h = boxValues[boxBase + 3],
                .score = scoreClassValues[scoreClassBase + 0],
                .classId = std::bit_cast<std::uint32_t>(scoreClassValues[scoreClassBase + 1]),
            });
        }

        readbackFinalScoreClass->Unmap(0, nullptr);
        readbackFinalBox->Unmap(0, nullptr);
        return results;
    }

    void setScoreThreshold(float value) { config.scoreThreshold = value; }

    void reset() {
        initialized = false;
        dxContext = nullptr;
        anchorCount = 0;
        attributeCount = 0;
        outputElementCount = 0;
        activeClassCount = 0;
        classStartIndex = 0;
        useObjectness = false;
        inputResolution = Resolution{.width = 0, .height = 0};

        if (fenceEvent != nullptr) {
            CloseHandle(fenceEvent);
            fenceEvent = nullptr;
        }
        fenceValue = 0;

        commandAllocator = nullptr;
        commandList = nullptr;
        completionFence = nullptr;
        descriptorHeap = nullptr;
        descriptorSize = 0;

        rootSignature = nullptr;
        decodePipelineState = nullptr;
        nmsPipelineState = nullptr;
        compactPipelineState = nullptr;

        candidateBox = nullptr;
        candidateScoreClass = nullptr;
        candidateCount = nullptr;
        suppressed = nullptr;
        finalBox = nullptr;
        finalScoreClass = nullptr;
        finalCount = nullptr;

        readbackFinalBox = nullptr;
        readbackFinalScoreClass = nullptr;
        readbackFinalCount = nullptr;

        candidateBoxState = D3D12_RESOURCE_STATE_COMMON;
        candidateScoreClassState = D3D12_RESOURCE_STATE_COMMON;
        candidateCountState = D3D12_RESOURCE_STATE_COMMON;
        suppressedState = D3D12_RESOURCE_STATE_COMMON;
        finalBoxState = D3D12_RESOURCE_STATE_COMMON;
        finalScoreClassState = D3D12_RESOURCE_STATE_COMMON;
        finalCountState = D3D12_RESOURCE_STATE_COMMON;
    }

  private:
    auto validateInitInputs(nrx::gfx::DxContext* context,
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

    auto configureOutputShape(std::span<const int64_t> outputShape)
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

    auto initializePipelineResources(ID3D12Device* d12Device)
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
        const auto nmsShaderResult = compileShader(
            shaders::kPostNmsCsHlslSource, shaders::kPostNmsCsHlslSourceLength, "post_nms_cs.hlsl");
        if (!nmsShaderResult) {
            return std::unexpected(nmsShaderResult.error());
        }
        const auto compactShaderResult =
            compileShader(shaders::kPostCompactCsHlslSource,
                          shaders::kPostCompactCsHlslSourceLength, "post_compact_cs.hlsl");
        if (!compactShaderResult) {
            return std::unexpected(compactShaderResult.error());
        }

        const auto decodePsoResult = createComputePipelineState(d12Device, rootSignature.get(),
                                                                decodeShaderResult.value().get());
        if (!decodePsoResult) {
            return std::unexpected(decodePsoResult.error());
        }
        decodePipelineState = decodePsoResult.value();

        const auto nmsPsoResult = createComputePipelineState(d12Device, rootSignature.get(),
                                                             nmsShaderResult.value().get());
        if (!nmsPsoResult) {
            return std::unexpected(nmsPsoResult.error());
        }
        nmsPipelineState = nmsPsoResult.value();

        const auto compactPsoResult = createComputePipelineState(d12Device, rootSignature.get(),
                                                                 compactShaderResult.value().get());
        if (!compactPsoResult) {
            return std::unexpected(compactPsoResult.error());
        }
        compactPipelineState = compactPsoResult.value();

        return {};
    }

    auto initializeDescriptorHeap(ID3D12Device* d12Device) -> std::expected<void, InferenceError> {
        D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
        heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDescription.NumDescriptors = DescriptorTableLayout::kDescriptorCount;
        heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        const HRESULT createHeapHr =
            d12Device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(descriptorHeap.put()));
        NRX_DX_CHECK(createHeapHr, "Postprocessor descriptor heap creation failed",
                     InferenceError::PostprocessFailed);
        descriptorSize =
            d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return {};
    }

    auto initializeBuffersAndDescriptors(ID3D12Device* d12Device)
        -> std::expected<void, InferenceError> {
        const auto candidateElementCount = static_cast<std::uint64_t>(anchorCount);
        const auto maxDetectionCount = static_cast<std::uint64_t>(config.maxDetections);

        struct BufferSpec {
            winrt::com_ptr<ID3D12Resource>* resource;
            std::uint32_t stride;
            std::uint64_t elementCount;
            D3D12_RESOURCE_FLAGS flags;
            D3D12_HEAP_TYPE heapType;
        };

        const std::array bufferSpecs{
            BufferSpec{&candidateBox, sizeof(float) * 4, candidateElementCount,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT},
            BufferSpec{&candidateScoreClass, sizeof(float) * 2, candidateElementCount,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT},
            BufferSpec{&candidateCount, sizeof(std::uint32_t), 1,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT},
            BufferSpec{&suppressed, sizeof(std::uint32_t), candidateElementCount,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT},
            BufferSpec{&finalBox, sizeof(float) * 4, maxDetectionCount,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT},
            BufferSpec{&finalScoreClass, sizeof(float) * 2, maxDetectionCount,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT},
            BufferSpec{&finalCount, sizeof(std::uint32_t), 1,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT},
            BufferSpec{&readbackFinalBox, sizeof(float) * 4, maxDetectionCount,
                       D3D12_RESOURCE_FLAG_NONE, D3D12_HEAP_TYPE_READBACK},
            BufferSpec{&readbackFinalScoreClass, sizeof(float) * 2, maxDetectionCount,
                       D3D12_RESOURCE_FLAG_NONE, D3D12_HEAP_TYPE_READBACK},
            BufferSpec{&readbackFinalCount, sizeof(std::uint32_t), 1, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_HEAP_TYPE_READBACK},
        };

        for (const auto& spec : bufferSpecs) {
            if (!createBuffer(d12Device, *spec.resource, spec.stride, spec.elementCount, spec.flags,
                              spec.heapType)) {
                reset();
                return std::unexpected(InferenceError::PostprocessFailed);
            }
        }

        createStaticDescriptors(d12Device);
        return {};
    }

    auto initializeCommandAndSyncObjects(ID3D12Device* d12Device)
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
        NRX_DX_CHECK(createFenceHr, "Postprocessor fence creation failed",
                     InferenceError::PostprocessFailed);
        fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (fenceEvent == nullptr) {
            return std::unexpected(InferenceError::PostprocessFailed);
        }

        return {};
    }

    auto createBuffer(ID3D12Device* d12Device, winrt::com_ptr<ID3D12Resource>& resource,
                      std::uint32_t stride, std::uint64_t elementCount, D3D12_RESOURCE_FLAGS flags,
                      D3D12_HEAP_TYPE heapType) -> bool {
        const auto sizeBytes = static_cast<std::uint64_t>(stride) * elementCount;
        const auto bufferResult =
            nrx::utils::DxHelper::createBuffer(d12Device, sizeBytes, flags, heapType);
        if (!bufferResult) {
            NRX_ERROR("Postprocessor buffer creation failed: {}",
                      nrx::utils::DxHelper::getErrorString(bufferResult.error()));
            return false;
        }

        resource = bufferResult.value();
        return true;
    }

    auto createStructuredSrv(ID3D12Device* d12Device, ID3D12Resource* resource,
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
        d12Device->CreateShaderResourceView(resource, &description,
                                            descriptorCpuHandle(viewSpec.descriptorIndex));
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    }

    auto createStructuredUav(ID3D12Device* d12Device, ID3D12Resource* resource,
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

    auto createStaticDescriptors(ID3D12Device* d12Device) -> void {
        createStructuredSrv(d12Device, candidateBox.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kSrvCandidateBox,
                                .stride = sizeof(float) * 4,
                                .numElements = anchorCount,
                            });
        createStructuredSrv(d12Device, candidateScoreClass.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kSrvCandidateScoreClass,
                                .stride = sizeof(float) * 2,
                                .numElements = anchorCount,
                            });
        createStructuredSrv(d12Device, candidateCount.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kSrvCandidateCount,
                                .stride = sizeof(std::uint32_t),
                                .numElements = 1,
                            });
        createStructuredSrv(d12Device, suppressed.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kSrvSuppressed,
                                .stride = sizeof(std::uint32_t),
                                .numElements = anchorCount,
                            });

        createStructuredUav(d12Device, candidateBox.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kUavCandidateBox,
                                .stride = sizeof(float) * 4,
                                .numElements = anchorCount,
                            });
        createStructuredUav(d12Device, candidateScoreClass.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kUavCandidateScoreClass,
                                .stride = sizeof(float) * 2,
                                .numElements = anchorCount,
                            });
        createStructuredUav(d12Device, candidateCount.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kUavCandidateCount,
                                .stride = sizeof(std::uint32_t),
                                .numElements = 1,
                            });
        createStructuredUav(d12Device, suppressed.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kUavSuppressed,
                                .stride = sizeof(std::uint32_t),
                                .numElements = anchorCount,
                            });
        createStructuredUav(d12Device, finalBox.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kUavFinalBox,
                                .stride = sizeof(float) * 4,
                                .numElements = config.maxDetections,
                            });
        createStructuredUav(d12Device, finalScoreClass.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kUavFinalScoreClass,
                                .stride = sizeof(float) * 2,
                                .numElements = config.maxDetections,
                            });
        createStructuredUav(d12Device, finalCount.get(),
                            StructuredViewSpec{
                                .descriptorIndex = DescriptorTableLayout::kUavFinalCount,
                                .stride = sizeof(std::uint32_t),
                                .numElements = 1,
                            });
    }

    auto addUavBarrier(ID3D12Resource* resource) -> void {
        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.UAV.pResource = resource;
        commandList->ResourceBarrier(1, &barrier);
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    }

    auto clearCounterUav(ID3D12Resource* resource, UINT descriptorIndex)
        -> std::expected<void, InferenceError> {
        if (resource == nullptr || commandList == nullptr || descriptorHeap == nullptr) {
            return std::unexpected(InferenceError::PostprocessFailed);
        }

        const std::array<UINT, 4> clearValues = {0, 0, 0, 0};
        const auto gpuHandle = descriptorGpuHandle(descriptorIndex);
        const auto cpuHandle = descriptorCpuHandle(descriptorIndex);

        commandList->ClearUnorderedAccessViewUint(gpuHandle, cpuHandle, resource,
                                                  clearValues.data(), 0, nullptr);
        addUavBarrier(resource);
        return {};
    }

    struct TransitionRequest {
        ID3D12Resource* resource;
        D3D12_RESOURCE_STATES* trackedState;
    };

    auto transitionResources(std::initializer_list<TransitionRequest> requests,
                             D3D12_RESOURCE_STATES targetState) -> void {
        if (commandList == nullptr) {
            return;
        }

        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        barriers.reserve(requests.size());

        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        for (const auto& request : requests) {
            if (request.resource == nullptr || request.trackedState == nullptr ||
                *request.trackedState == targetState) {
                continue;
            }

            D3D12_RESOURCE_BARRIER transitionBarrier{};
            transitionBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            transitionBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            transitionBarrier.Transition.pResource = request.resource;
            transitionBarrier.Transition.StateBefore = *request.trackedState;
            transitionBarrier.Transition.StateAfter = targetState;
            transitionBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barriers.push_back(transitionBarrier);
            *request.trackedState = targetState;
        }

        if (!barriers.empty()) {
            commandList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
        }
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    }

    [[nodiscard]] auto descriptorCpuHandle(UINT descriptorIndex) const
        -> D3D12_CPU_DESCRIPTOR_HANDLE {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(),
                                             static_cast<INT>(descriptorIndex),
                                             static_cast<INT>(descriptorSize));
    }

    [[nodiscard]] auto descriptorGpuHandle(UINT descriptorIndex) const
        -> D3D12_GPU_DESCRIPTOR_HANDLE {
        return CD3DX12_GPU_DESCRIPTOR_HANDLE(descriptorHeap->GetGPUDescriptorHandleForHeapStart(),
                                             static_cast<INT>(descriptorIndex),
                                             static_cast<INT>(descriptorSize));
    }

    Config config{};
    bool initialized{false};
    nrx::gfx::DxContext* dxContext{nullptr};

    Resolution inputResolution{.width = 0, .height = 0};
    std::uint32_t anchorCount{0};
    std::uint32_t attributeCount{0};
    std::uint64_t outputElementCount{0};
    DetectionLayout layout{DetectionLayout::AttributeMajor};
    bool useObjectness{false};
    std::uint32_t classStartIndex{0};
    std::uint32_t activeClassCount{0};

    winrt::com_ptr<ID3D12RootSignature> rootSignature;
    winrt::com_ptr<ID3D12PipelineState> decodePipelineState;
    winrt::com_ptr<ID3D12PipelineState> nmsPipelineState;
    winrt::com_ptr<ID3D12PipelineState> compactPipelineState;

    winrt::com_ptr<ID3D12DescriptorHeap> descriptorHeap;
    UINT descriptorSize{0};

    winrt::com_ptr<ID3D12CommandAllocator> commandAllocator;
    winrt::com_ptr<ID3D12GraphicsCommandList> commandList;
    winrt::com_ptr<ID3D12Fence> completionFence;
    HANDLE fenceEvent{nullptr};
    std::uint64_t fenceValue{0};

    winrt::com_ptr<ID3D12Resource> candidateBox;
    winrt::com_ptr<ID3D12Resource> candidateScoreClass;
    winrt::com_ptr<ID3D12Resource> candidateCount;
    winrt::com_ptr<ID3D12Resource> suppressed;
    winrt::com_ptr<ID3D12Resource> finalBox;
    winrt::com_ptr<ID3D12Resource> finalScoreClass;
    winrt::com_ptr<ID3D12Resource> finalCount;

    winrt::com_ptr<ID3D12Resource> readbackFinalBox;
    winrt::com_ptr<ID3D12Resource> readbackFinalScoreClass;
    winrt::com_ptr<ID3D12Resource> readbackFinalCount;

    D3D12_RESOURCE_STATES candidateBoxState{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES candidateScoreClassState{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES candidateCountState{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES suppressedState{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES finalBoxState{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES finalScoreClassState{D3D12_RESOURCE_STATE_COMMON};
    D3D12_RESOURCE_STATES finalCountState{D3D12_RESOURCE_STATE_COMMON};
};

Postprocessor::Postprocessor() : impl(std::make_unique<Impl>(Config{})) {}

Postprocessor::Postprocessor(Config config) : impl(std::make_unique<Impl>(config)) {}

Postprocessor::~Postprocessor() = default;

auto Postprocessor::init(nrx::gfx::DxContext* dxContext, std::span<const int64_t> outputShape,
                         Resolution inputResolution) -> std::expected<void, InferenceError> {
    return impl->init(dxContext, outputShape, inputResolution);
}

auto Postprocessor::dispatch(ID3D12Resource* rawOutputResource, D3D12_RESOURCE_STATES currentState)
    -> std::expected<void, InferenceError> {
    return impl->dispatch(rawOutputResource, currentState);
}

auto Postprocessor::readbackFinalResults() -> std::expected<DetectionResults, InferenceError> {
    return impl->readbackFinalResults();
}

void Postprocessor::setScoreThreshold(float value) { impl->setScoreThreshold(value); }

void Postprocessor::reset() { impl->reset(); }

} // namespace nrx::inference
