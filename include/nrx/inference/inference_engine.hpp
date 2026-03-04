#pragma once

#include <expected>
#include <filesystem>
#include <memory>

#include "nrx/gfx/dx_types.hpp"
#include "nrx/inference/types.hpp"

struct ID3D12Resource;

namespace nrx::gfx {
class DxContext;
}

namespace nrx::inference {

class InferenceEngine {
  public:
    InferenceEngine();
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    auto operator=(const InferenceEngine&) -> InferenceEngine& = delete;
    InferenceEngine(InferenceEngine&&) = delete;
    auto operator=(InferenceEngine&&) -> InferenceEngine& = delete;

    auto init(nrx::gfx::DxContext* dxContext, const std::filesystem::path& modelPath)
        -> std::expected<void, InferenceError>;

    auto execute(ID3D12Resource* inputTexture, D3D12_RESOURCE_STATES currentState)
        -> std::expected<DetectionResults, InferenceError>;

    auto update(const std::filesystem::path& modelPath, float confidenceThreshold) -> bool;
    auto reinitialize() -> bool;
    void setScoreThreshold(float value);
    void reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::inference
