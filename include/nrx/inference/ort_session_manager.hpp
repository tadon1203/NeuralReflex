#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>

#include "nrx/inference/types.hpp"

struct ID3D12Resource;

namespace nrx::gfx {
class DxContext;
}

namespace nrx::inference {

enum class InferenceError : std::uint8_t;

class OrtSessionManager {
  public:
    OrtSessionManager();
    ~OrtSessionManager();

    OrtSessionManager(const OrtSessionManager&) = delete;
    auto operator=(const OrtSessionManager&) -> OrtSessionManager& = delete;
    OrtSessionManager(OrtSessionManager&&) = delete;
    auto operator=(OrtSessionManager&&) -> OrtSessionManager& = delete;

    auto init(nrx::gfx::DxContext* dxContext, const std::filesystem::path& modelPath)
        -> std::expected<void, InferenceError>;

    auto run(ID3D12Resource* inputTensorResource)
        -> std::expected<ID3D12Resource*, InferenceError>;

    [[nodiscard]] auto getInputResolution() const -> Resolution;
    [[nodiscard]] auto outputShape() const -> std::span<const int64_t>;

    void reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::inference
