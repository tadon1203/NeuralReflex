#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include "nrx/gfx/dx_types.hpp"
#include "nrx/inference/types.hpp"

struct ID3D12Resource;

namespace nrx::gfx {
class DxContext;
}

namespace nrx::inference {

class Postprocessor {
  public:
    struct Config {
        float scoreThreshold{0.45F};
        float nmsIouThreshold{0.45F};
        std::uint32_t classCount{1};
        std::uint32_t maxDetections{256};
    };

    Postprocessor();
    explicit Postprocessor(Config config);
    ~Postprocessor();

    Postprocessor(const Postprocessor&) = delete;
    auto operator=(const Postprocessor&) -> Postprocessor& = delete;
    Postprocessor(Postprocessor&&) = delete;
    auto operator=(Postprocessor&&) -> Postprocessor& = delete;

    auto init(nrx::gfx::DxContext* dxContext, std::span<const int64_t> outputShape,
              Resolution inputResolution) -> std::expected<void, InferenceError>;
    auto dispatch(ID3D12Resource* rawOutputResource, D3D12_RESOURCE_STATES currentState)
        -> std::expected<void, InferenceError>;
    auto readbackFinalResults() -> std::expected<DetectionResults, InferenceError>;
    void setScoreThreshold(float value);
    void reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::inference
