#pragma once

#include <cstdint>
#include <expected>
#include <memory>

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

class ImagePreprocessor {
  public:
    ImagePreprocessor();
    ~ImagePreprocessor();

    ImagePreprocessor(const ImagePreprocessor&) = delete;
    auto operator=(const ImagePreprocessor&) -> ImagePreprocessor& = delete;
    ImagePreprocessor(ImagePreprocessor&&) = delete;
    auto operator=(ImagePreprocessor&&) -> ImagePreprocessor& = delete;

    auto init(nrx::gfx::DxContext* dxContext) -> std::expected<void, InferenceError>;
    [[nodiscard]] auto preprocess(ID3D12Resource* inputTexture,
                                  D3D12_RESOURCE_STATES currentState,
                                  D3D12_RESOURCE_STATES nextState)
        -> std::expected<ID3D12Resource*, InferenceError>;
    void reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::inference
