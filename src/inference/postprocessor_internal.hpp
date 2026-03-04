#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <span>

#include <Windows.h>
#include <directx/d3d12.h>
#include <winrt/base.h>

#include "nrx/inference/postprocessor.hpp"

namespace nrx::gfx {
class DxContext;
}

namespace nrx::inference {

class Postprocessor::Impl {
  public:
    static constexpr std::size_t kMinDetAttributes = 5;
    static constexpr UINT kThreadGroupSize = 256;
    static constexpr UINT kRootConstantCount = 12;

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
        static constexpr UINT kMaxTransitionBatchSize = kUavCount + 1;
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

    struct TransitionRequest {
        ID3D12Resource* resource;
        D3D12_RESOURCE_STATES* trackedState;
    };

    enum class BufferKind : std::uint8_t {
        Uav,
        Readback,
    };

    struct BufferSpec {
        winrt::com_ptr<ID3D12Resource>* resource;
        std::uint64_t sizeBytes;
        BufferKind kind;
    };

    explicit Impl(Config configValue);

    auto init(nrx::gfx::DxContext* context, std::span<const int64_t> outputShape,
              Resolution inputResolutionValue) -> std::expected<void, InferenceError>;
    auto dispatch(ID3D12Resource* rawOutputResource, D3D12_RESOURCE_STATES currentState)
        -> std::expected<void, InferenceError>;
    auto readbackFinalResults() -> std::expected<DetectionResults, InferenceError>;
    void setScoreThreshold(float value);
    void reset();

  private:
    auto validateInitInputs(nrx::gfx::DxContext* context,
                            std::span<const int64_t> outputShape) const
        -> std::expected<ID3D12Device*, InferenceError>;
    auto configureOutputShape(std::span<const int64_t> outputShape)
        -> std::expected<void, InferenceError>;
    auto initializePipelineResources(ID3D12Device* d12Device)
        -> std::expected<void, InferenceError>;
    auto initializeDescriptorHeap(ID3D12Device* d12Device) -> std::expected<void, InferenceError>;
    auto initializeBuffersAndDescriptors(ID3D12Device* d12Device)
        -> std::expected<void, InferenceError>;
    auto initializeCommandAndSyncObjects(ID3D12Device* d12Device)
        -> std::expected<void, InferenceError>;

    auto createStructuredSrv(ID3D12Device* d12Device, ID3D12Resource* resource,
                             const StructuredViewSpec& viewSpec) -> void;
    auto createStructuredUav(ID3D12Device* d12Device, ID3D12Resource* resource,
                             const StructuredViewSpec& viewSpec) -> void;
    auto createStaticDescriptors(ID3D12Device* d12Device) -> void;

    auto runDecodeStage(UINT groupCount) -> void;
    auto runNmsStage(UINT groupCount) -> void;
    auto runCompactStage(UINT groupCount) -> void;
    auto copyReadbackResources() -> void;
    auto addUavBarrier(ID3D12Resource* resource) -> void;
    auto clearCounterUav(ID3D12Resource* resource, UINT descriptorIndex)
        -> std::expected<void, InferenceError>;
    auto transitionResources(std::initializer_list<TransitionRequest> requests,
                             D3D12_RESOURCE_STATES targetState) -> void;

    [[nodiscard]] auto descriptorCpuHandle(UINT descriptorIndex) const
        -> D3D12_CPU_DESCRIPTOR_HANDLE;
    [[nodiscard]] auto descriptorGpuHandle(UINT descriptorIndex) const
        -> D3D12_GPU_DESCRIPTOR_HANDLE;

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

} // namespace nrx::inference
