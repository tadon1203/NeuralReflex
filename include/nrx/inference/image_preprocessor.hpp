#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "nrx/gfx/dx_types.hpp"

struct ID3D12Resource;

namespace nrx::gfx {
class DxContext;
}

namespace nrx::inference {

enum class InferenceError : std::uint8_t;
struct ResourceTransition;

struct PreprocessOutput {
    ID3D12Resource* resource{nullptr};
    D3D12_RESOURCE_STATES currentState =
        static_cast<D3D12_RESOURCE_STATES>(0); // NOLINT(readability-identifier-naming)
};

class ImagePreprocessor {
  public:
    ImagePreprocessor();
    ~ImagePreprocessor();

    ImagePreprocessor(const ImagePreprocessor&) = delete;
    auto operator=(const ImagePreprocessor&) -> ImagePreprocessor& = delete;
    ImagePreprocessor(ImagePreprocessor&&) = delete;
    auto operator=(ImagePreprocessor&&) -> ImagePreprocessor& = delete;

    auto init(nrx::gfx::DxContext* dxContext) -> std::expected<void, InferenceError>;
    [[nodiscard]] auto preprocess(ID3D12Resource* inputTexture, const ResourceTransition& transition)
        -> std::expected<PreprocessOutput, InferenceError>;
    void reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::inference
