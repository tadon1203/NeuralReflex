#include "nrx/gfx/screen_capturer.hpp"

#include <algorithm>
#include <expected>
#include <utility>
#include <vector>

#include <Windows.h>
#include <d3d11_4.h>
#include <dxgi.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/base.h>

#include "nrx/gfx/dx_context.hpp"
#include "nrx/utils/dx_helper.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::gfx {

auto captureErrorToString(CaptureError error) -> std::string_view {
    switch (error) {
    case CaptureError::InvalidContext:
        return "InvalidContext";
    case CaptureError::DxContextNotInitialized:
        return "DxContextNotInitialized";
    case CaptureError::GraphicsCaptureNotSupported:
        return "GraphicsCaptureNotSupported";
    case CaptureError::MonitorEnumerationFailed:
        return "MonitorEnumerationFailed";
    case CaptureError::NoMonitors:
        return "NoMonitors";
    case CaptureError::NoPrimaryMonitor:
        return "NoPrimaryMonitor";
    case CaptureError::DisplayIndexOutOfRange:
        return "DisplayIndexOutOfRange";
    case CaptureError::DxgiDeviceQueryFailed:
        return "DxgiDeviceQueryFailed";
    case CaptureError::WinRtDeviceCreationFailed:
        return "WinRtDeviceCreationFailed";
    case CaptureError::CaptureItemCreationFailed:
        return "CaptureItemCreationFailed";
    case CaptureError::NotInitialized:
        return "NotInitialized";
    case CaptureError::AlreadyCapturing:
        return "AlreadyCapturing";
    case CaptureError::StartCaptureFailed:
        return "StartCaptureFailed";
    case CaptureError::NotCapturing:
        return "NotCapturing";
    case CaptureError::AcquireFrameFailed:
        return "AcquireFrameFailed";
    case CaptureError::NoNewFrameAvailable:
        return "NoNewFrameAvailable";
    case CaptureError::NullFrameSurface:
        return "NullFrameSurface";
    case CaptureError::SurfaceAccessQueryFailed:
        return "SurfaceAccessQueryFailed";
    case CaptureError::TextureQueryFailed:
        return "TextureQueryFailed";
    }

    return "UnknownCaptureError";
}

namespace {

struct MonitorInfo {
    HMONITOR handle{nullptr};
};

auto CALLBACK enumMonitorsProc(HMONITOR monitor, HDC deviceContext, LPRECT clippingRect,
                               LPARAM callbackData) -> BOOL {
    static_cast<void>(deviceContext);
    static_cast<void>(clippingRect);

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* monitors = reinterpret_cast<std::vector<MonitorInfo>*>(callbackData);
    if (monitors == nullptr) {
        return FALSE;
    }

    monitors->push_back(MonitorInfo{.handle = monitor});
    return TRUE;
}

auto enumerateMonitors() -> std::expected<std::vector<MonitorInfo>, CaptureError> {
    std::vector<MonitorInfo> monitors;
    const BOOL enumOk = EnumDisplayMonitors(
        nullptr, nullptr, enumMonitorsProc,
        reinterpret_cast<LPARAM>(&monitors)); // NOLINT(performance-no-int-to-ptr)

    if (enumOk == 0) {
        NRX_ERROR("Failed to enumerate monitors: {}",
                  nrx::utils::DxHelper::getErrorString(
                      HRESULT_FROM_WIN32(static_cast<unsigned int>(GetLastError()))));
        return std::unexpected(CaptureError::MonitorEnumerationFailed);
    }

    if (monitors.empty()) {
        return std::unexpected(CaptureError::NoMonitors);
    }

    return monitors;
}

} // namespace

class ScreenCapturer::Impl {
  public:
    explicit Impl(DxContext* context) : dxContext(context) {}
    ~Impl() { stop(); }

    Impl(const Impl&) = delete;
    auto operator=(const Impl&) -> Impl& = delete;
    Impl(Impl&&) = delete;
    auto operator=(Impl&&) -> Impl& = delete;

    auto init(std::int32_t displayIndex) -> std::expected<void, CaptureError> {
        stop();
        resetInitState();

        if (dxContext == nullptr) {
            return std::unexpected(CaptureError::InvalidContext);
        }

        if (dxContext->getD11Device() == nullptr) {
            return std::unexpected(CaptureError::DxContextNotInitialized);
        }

        if (!winrt::Windows::Graphics::Capture::GraphicsCaptureSession::IsSupported()) {
            return std::unexpected(CaptureError::GraphicsCaptureNotSupported);
        }

        d11Device.copy_from(dxContext->getD11Device());

        winrt::com_ptr<IDXGIDevice> dxgiDevice;
        const HRESULT dxgiHr = d11Device->QueryInterface(IID_PPV_ARGS(dxgiDevice.put()));
        NRX_DX_CHECK(dxgiHr, "Failed to query IDXGIDevice from D3D11 device",
                     CaptureError::DxgiDeviceQueryFailed);

        winrt::com_ptr<IInspectable> inspectableDevice;
        const HRESULT interopHr =
            CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.get(), inspectableDevice.put());
        NRX_DX_CHECK(interopHr, "Failed to create WinRT Direct3D11 device",
                     CaptureError::WinRtDeviceCreationFailed);

        d3dDevice =
            inspectableDevice.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();

        const auto monitorsResult = enumerateMonitors();
        if (!monitorsResult) {
            return std::unexpected(monitorsResult.error());
        }

        const auto& monitors = monitorsResult.value();
        if (displayIndex < 0 || static_cast<std::size_t>(displayIndex) >= monitors.size()) {
            NRX_WARN("Display index {} is out of range. Available displays: {}.", displayIndex,
                     monitors.size());
            return std::unexpected(CaptureError::DisplayIndexOutOfRange);
        }

        selectedMonitor = monitors[static_cast<std::size_t>(displayIndex)].handle;
        selectedDisplayIndex = displayIndex;

        const auto captureInterop =
            winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem,
                                          IGraphicsCaptureItemInterop>();

        winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{nullptr};
        const HRESULT itemHr = captureInterop->CreateForMonitor(
            selectedMonitor,
            winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
            winrt::put_abi(item));
        NRX_DX_CHECK(itemHr, "Failed to create GraphicsCaptureItem for selected monitor",
                     CaptureError::CaptureItemCreationFailed);

        captureItem = item;
        initialized = true;

        NRX_INFO("ScreenCapturer initialized for display index {}.", selectedDisplayIndex);
        return {};
    }

    auto reconfigure(std::int32_t displayIndex) -> bool {
        const auto previousDisplayIndex = selectedDisplayIndex;
        const bool wasCapturing = capturing;

        if (const auto initResult = init(displayIndex); !initResult) {
            NRX_WARN("ScreenCapturer reconfigure failed for display index {}: {}", displayIndex,
                     captureErrorToString(initResult.error()));
            if (previousDisplayIndex >= 0) {
                if (const auto restoreInitResult = init(previousDisplayIndex); !restoreInitResult) {
                    NRX_CRITICAL("ScreenCapturer rollback init failed for display index {}: {}",
                                 previousDisplayIndex,
                                 captureErrorToString(restoreInitResult.error()));
                    return false;
                }
                if (wasCapturing) {
                    if (const auto restoreStartResult = start(); !restoreStartResult) {
                        NRX_CRITICAL(
                            "ScreenCapturer rollback start failed for display index {}: {}",
                            previousDisplayIndex,
                            captureErrorToString(restoreStartResult.error()));
                        return false;
                    }
                }
            }
            return false;
        }

        if (wasCapturing) {
            if (const auto startResult = start(); !startResult) {
                NRX_WARN("ScreenCapturer reconfigure start failed for display index {}: {}",
                         displayIndex, captureErrorToString(startResult.error()));

                if (previousDisplayIndex >= 0) {
                    if (const auto restoreInitResult = init(previousDisplayIndex);
                        !restoreInitResult) {
                        NRX_CRITICAL("ScreenCapturer rollback init failed for display index {}: {}",
                                     previousDisplayIndex,
                                     captureErrorToString(restoreInitResult.error()));
                        return false;
                    }
                    if (const auto restoreStartResult = start(); !restoreStartResult) {
                        NRX_CRITICAL(
                            "ScreenCapturer rollback start failed for display index {}: {}",
                            previousDisplayIndex,
                            captureErrorToString(restoreStartResult.error()));
                        return false;
                    }
                }
                return false;
            }
        }

        return true;
    }

    auto start() -> std::expected<void, CaptureError> {
        if (!initialized) {
            return std::unexpected(CaptureError::NotInitialized);
        }
        if (capturing) {
            return std::unexpected(CaptureError::AlreadyCapturing);
        }

        try {
            framePool =
                winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
                    d3dDevice,
                    winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
                    2, captureItem.Size());
            captureSession = framePool.CreateCaptureSession(captureItem);
            captureSession.IsBorderRequired(false);
            captureSession.StartCapture();
        } catch (const winrt::hresult_error& e) {
            NRX_ERROR("Failed to start screen capture: {}", winrt::to_string(e.message()));
            return std::unexpected(CaptureError::StartCaptureFailed);
        }

        capturing = true;
        NRX_INFO("ScreenCapturer started.");
        return {};
    }

    void stop() {
        if (captureSession) {
            captureSession.Close();
            captureSession = nullptr;
        }

        if (framePool) {
            framePool.Close();
            framePool = nullptr;
        }

        latestTexture = nullptr;
        capturing = false;
    }

    auto acquireNextFrame() -> std::expected<ID3D11Texture2D*, CaptureError> {
        if (!initialized) {
            return std::unexpected(CaptureError::NotInitialized);
        }
        if (!capturing) {
            return std::unexpected(CaptureError::NotCapturing);
        }

        winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame frame{nullptr};
        try {
            frame = framePool.TryGetNextFrame();
        } catch (const winrt::hresult_error& e) {
            NRX_ERROR("Failed to acquire next frame: {}", winrt::to_string(e.message()));
            return std::unexpected(CaptureError::AcquireFrameFailed);
        }

        if (!frame) {
            return std::unexpected(CaptureError::NoNewFrameAvailable);
        }

        const auto surface = frame.Surface();
        if (!surface) {
            return std::unexpected(CaptureError::NullFrameSurface);
        }

        winrt::com_ptr<IInspectable> inspectableSurface;
        winrt::copy_from_abi(inspectableSurface, winrt::get_abi(surface));

        winrt::com_ptr<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>
            surfaceAccess;
        try {
            surfaceAccess =
                inspectableSurface
                    .as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        } catch (const winrt::hresult_error& e) {
            NRX_ERROR("Failed to query IDirect3DDxgiInterfaceAccess: {}",
                      winrt::to_string(e.message()));
            return std::unexpected(CaptureError::SurfaceAccessQueryFailed);
        }

        winrt::com_ptr<ID3D11Texture2D> texture;
        const HRESULT textureHr =
            surfaceAccess->GetInterface(__uuidof(ID3D11Texture2D), texture.put_void());
        NRX_DX_CHECK(textureHr, "Failed to get ID3D11Texture2D from frame surface",
                     CaptureError::TextureQueryFailed);

        latestTexture = std::move(texture);
        return latestTexture.get();
    }

  private:
    void resetInitState() {
        initialized = false;
        selectedMonitor = nullptr;
        selectedDisplayIndex = -1;
        captureItem = nullptr;
        d3dDevice = nullptr;
        d11Device = nullptr;
    }

    DxContext* dxContext{nullptr};

    bool initialized{false};
    bool capturing{false};

    HMONITOR selectedMonitor{nullptr};
    std::int32_t selectedDisplayIndex{-1};

    winrt::com_ptr<ID3D11Device5> d11Device;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice d3dDevice{nullptr};

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem captureItem{nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool framePool{nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession captureSession{nullptr};

    winrt::com_ptr<ID3D11Texture2D> latestTexture;
};

ScreenCapturer::ScreenCapturer(DxContext* ctx) : impl(std::make_unique<Impl>(ctx)) {}

ScreenCapturer::~ScreenCapturer() = default;

auto ScreenCapturer::init(std::int32_t displayIndex) -> std::expected<void, CaptureError> {
    return impl->init(displayIndex);
}

auto ScreenCapturer::reconfigure(std::int32_t displayIndex) -> bool {
    return impl->reconfigure(displayIndex);
}

auto ScreenCapturer::start() -> std::expected<void, CaptureError> { return impl->start(); }

void ScreenCapturer::stop() { impl->stop(); }

auto ScreenCapturer::acquireNextFrame() -> std::expected<ID3D11Texture2D*, CaptureError> {
    return impl->acquireNextFrame();
}

} // namespace nrx::gfx
