#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>

#include "nrx/inference/types.hpp"

struct ID3D12Resource;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmicrosoft-enum-forward-reference"
#endif
using D3D12_RESOURCE_STATES = enum D3D12_RESOURCE_STATES; // NOLINT(readability-identifier-naming)
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace nrx::gfx {
class DxContext;
}

namespace nrx::inference {

enum class InferenceError : std::uint8_t;

struct OrtSessionOutput {
    ID3D12Resource* resource{nullptr};
    D3D12_RESOURCE_STATES currentState =
        static_cast<D3D12_RESOURCE_STATES>(0); // NOLINT(readability-identifier-naming)
};

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
        -> std::expected<OrtSessionOutput, InferenceError>;

    [[nodiscard]] auto getInputResolution() const -> Resolution;
    [[nodiscard]] auto outputShape() const -> std::span<const int64_t>;

    void reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::inference
