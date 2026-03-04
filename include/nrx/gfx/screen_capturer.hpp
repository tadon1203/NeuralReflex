#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string_view>

struct ID3D11Texture2D;

namespace nrx::gfx {

class DxContext;

enum class CaptureError : std::uint8_t {
    InvalidContext,
    DxContextNotInitialized,
    GraphicsCaptureNotSupported,
    MonitorEnumerationFailed,
    NoMonitors,
    NoPrimaryMonitor,
    DisplayIndexOutOfRange,
    DxgiDeviceQueryFailed,
    WinRtDeviceCreationFailed,
    CaptureItemCreationFailed,
    NotInitialized,
    AlreadyCapturing,
    StartCaptureFailed,
    NotCapturing,
    AcquireFrameFailed,
    NoNewFrameAvailable,
    NullFrameSurface,
    SurfaceAccessQueryFailed,
    TextureQueryFailed,
};

[[nodiscard]] auto captureErrorToString(CaptureError error) -> std::string_view;

class ScreenCapturer {
  public:
    explicit ScreenCapturer(DxContext* ctx);
    ~ScreenCapturer();

    ScreenCapturer(const ScreenCapturer&) = delete;
    auto operator=(const ScreenCapturer&) -> ScreenCapturer& = delete;
    ScreenCapturer(ScreenCapturer&&) = delete;
    auto operator=(ScreenCapturer&&) -> ScreenCapturer& = delete;

    auto init(std::int32_t displayIndex) -> std::expected<void, CaptureError>;
    auto reconfigure(std::int32_t displayIndex) -> bool;
    auto start() -> std::expected<void, CaptureError>;
    void stop();

    auto acquireNextFrame() -> std::expected<ID3D11Texture2D*, CaptureError>;

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::gfx
