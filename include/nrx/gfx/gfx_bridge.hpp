#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string_view>

struct ID3D11Texture2D;
struct ID3D12Resource;

namespace nrx::gfx {

class DxContext;

enum class BridgeError : std::uint8_t {
    InvalidArguments,
    DxContextNull,
    ResourceSharingFailed,
    SyncFailed,
    DeviceLost,
};

[[nodiscard]] auto bridgeErrorToString(BridgeError error) -> std::string_view;

class GfxBridge {
  public:
    explicit GfxBridge(DxContext* context);
    ~GfxBridge();

    GfxBridge(const GfxBridge&) = delete;
    auto operator=(const GfxBridge&) -> GfxBridge& = delete;
    GfxBridge(GfxBridge&&) = delete;
    auto operator=(GfxBridge&&) -> GfxBridge& = delete;

    auto registerTexture(ID3D11Texture2D* d11Texture)
        -> std::expected<ID3D12Resource*, BridgeError>;

    auto synchronize() -> std::expected<void, BridgeError>;

    void reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::gfx
