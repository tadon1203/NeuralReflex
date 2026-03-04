#include "nrx/inference/postprocessor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <memory>
#include <span>

#include <Windows.h>
#include <directx/d3d12.h>
#include <directx/d3dx12.h>

#include "nrx/gfx/dx_context.hpp"
#include "inference/postprocessor_internal.hpp"
#include "nrx/utils/dx_helper.hpp"

namespace nrx::inference {

auto Postprocessor::Impl::dispatch(ID3D12Resource* rawOutputResource,
                                   D3D12_RESOURCE_STATES currentState)
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
    d12Device->CreateShaderResourceView(rawOutputResource, &rawOutputSrv,
                                        descriptorCpuHandle(DescriptorTableLayout::kSrvRawOutput));
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)

    const HRESULT resetAllocatorHr = commandAllocator->Reset();
    NRX_DX_CHECK(resetAllocatorHr, "Postprocessor command allocator reset failed",
                 InferenceError::PostprocessFailed);
    const HRESULT resetListHr = commandList->Reset(commandAllocator.get(), decodePipelineState.get());
    NRX_DX_CHECK(resetListHr, "Postprocessor command list reset failed",
                 InferenceError::PostprocessFailed);

    const std::array<ID3D12DescriptorHeap*, 1> descriptorHeaps = {descriptorHeap.get()};
    commandList->SetDescriptorHeaps(static_cast<UINT>(descriptorHeaps.size()), descriptorHeaps.data());
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
    commandList->SetComputeRootDescriptorTable(1,
                                               descriptorGpuHandle(DescriptorTableLayout::kSrvRawOutput));
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
        const HRESULT waitEventHr = completionFence->SetEventOnCompletion(fenceValue, fenceEvent);
        NRX_DX_CHECK(waitEventHr, "Postprocessor wait event setup failed",
                     InferenceError::PostprocessFailed);
        WaitForSingleObject(fenceEvent, INFINITE);
    }

    return {};
}

auto Postprocessor::Impl::runDecodeStage(UINT groupCount) -> void {
    commandList->Dispatch(groupCount, 1, 1);
    addUavBarrier(candidateBox.get());
    addUavBarrier(candidateScoreClass.get());
    addUavBarrier(candidateCount.get());
}

auto Postprocessor::Impl::runNmsStage(UINT groupCount) -> void {
    commandList->SetPipelineState(nmsPipelineState.get());
    commandList->Dispatch(groupCount, 1, 1);
    addUavBarrier(suppressed.get());
}

auto Postprocessor::Impl::runCompactStage(UINT groupCount) -> void {
    commandList->Dispatch(groupCount, 1, 1);
    addUavBarrier(finalBox.get());
    addUavBarrier(finalScoreClass.get());
    addUavBarrier(finalCount.get());
}

auto Postprocessor::Impl::copyReadbackResources() -> void {
    commandList->CopyResource(readbackFinalBox.get(), finalBox.get());
    commandList->CopyResource(readbackFinalScoreClass.get(), finalScoreClass.get());
    commandList->CopyResource(readbackFinalCount.get(), finalCount.get());
}

auto Postprocessor::Impl::readbackFinalResults() -> std::expected<DetectionResults, InferenceError> {
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
    const auto scoreClassReadSize = static_cast<std::size_t>(finalCountValue) * sizeof(float) * 2;
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

void Postprocessor::Impl::setScoreThreshold(float value) { config.scoreThreshold = value; }

void Postprocessor::Impl::reset() {
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

auto Postprocessor::Impl::addUavBarrier(ID3D12Resource* resource) -> void {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.UAV.pResource = resource;
    commandList->ResourceBarrier(1, &barrier);
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
}

auto Postprocessor::Impl::clearCounterUav(ID3D12Resource* resource, UINT descriptorIndex)
    -> std::expected<void, InferenceError> {
    if (resource == nullptr || commandList == nullptr || descriptorHeap == nullptr) {
        return std::unexpected(InferenceError::PostprocessFailed);
    }

    const std::array<UINT, 4> clearValues = {0, 0, 0, 0};
    const auto gpuHandle = descriptorGpuHandle(descriptorIndex);
    const auto cpuHandle = descriptorCpuHandle(descriptorIndex);

    commandList->ClearUnorderedAccessViewUint(gpuHandle, cpuHandle, resource, clearValues.data(), 0,
                                              nullptr);
    addUavBarrier(resource);
    return {};
}

auto Postprocessor::Impl::transitionResources(std::initializer_list<TransitionRequest> requests,
                                              D3D12_RESOURCE_STATES targetState) -> void {
    if (commandList == nullptr) {
        return;
    }

    std::array<D3D12_RESOURCE_BARRIER, DescriptorTableLayout::kMaxTransitionBatchSize> barriers{};
    std::size_t barrierCount = 0;

    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    for (const auto& request : requests) {
        if (request.resource == nullptr || request.trackedState == nullptr ||
            *request.trackedState == targetState) {
            continue;
        }
        if (barrierCount >= barriers.size()) {
            NRX_ERROR("Postprocessor transition batch overflow: max={}, requested={}",
                      barriers.size(), requests.size());
            break;
        }

        D3D12_RESOURCE_BARRIER transitionBarrier{};
        transitionBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transitionBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        transitionBarrier.Transition.pResource = request.resource;
        transitionBarrier.Transition.StateBefore = *request.trackedState;
        transitionBarrier.Transition.StateAfter = targetState;
        transitionBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[barrierCount] = transitionBarrier;
        barrierCount += 1;
        *request.trackedState = targetState;
    }

    if (barrierCount > 0) {
        commandList->ResourceBarrier(static_cast<UINT>(barrierCount), barriers.data());
    }
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
}

[[nodiscard]] auto Postprocessor::Impl::descriptorCpuHandle(UINT descriptorIndex) const
    -> D3D12_CPU_DESCRIPTOR_HANDLE {
    return CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeap->GetCPUDescriptorHandleForHeapStart(),
                                         static_cast<INT>(descriptorIndex),
                                         static_cast<INT>(descriptorSize));
}

[[nodiscard]] auto Postprocessor::Impl::descriptorGpuHandle(UINT descriptorIndex) const
    -> D3D12_GPU_DESCRIPTOR_HANDLE {
    return CD3DX12_GPU_DESCRIPTOR_HANDLE(descriptorHeap->GetGPUDescriptorHandleForHeapStart(),
                                         static_cast<INT>(descriptorIndex),
                                         static_cast<INT>(descriptorSize));
}

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
