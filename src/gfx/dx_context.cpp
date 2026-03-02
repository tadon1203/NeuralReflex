#include "nrx/gfx/dx_context.hpp"

#include <atomic>
#include <iterator>

#include <Windows.h>
#include <d3d11.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_3.h>
#include <dxgi1_6.h>
#include <winrt/base.h>

#include "nrx/utils/dx_helper.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::gfx {

auto dxContextErrorToString(DxContextError error) -> std::string_view {
    switch (error) {
    case DxContextError::CreateFactoryFailed:
        return "CreateFactoryFailed";
    case DxContextError::FactoryNotInitialized:
        return "FactoryNotInitialized";
    case DxContextError::AdapterEnumerationFailed:
        return "AdapterEnumerationFailed";
    case DxContextError::NoHardwareAdapter:
        return "NoHardwareAdapter";
    case DxContextError::CreateD12DeviceFailed:
        return "CreateD12DeviceFailed";
    case DxContextError::D12DeviceNotInitialized:
        return "D12DeviceNotInitialized";
    case DxContextError::CreateD12QueueFailed:
        return "CreateD12QueueFailed";
    case DxContextError::CreateD11DeviceFailed:
        return "CreateD11DeviceFailed";
    case DxContextError::QueryD11AdvancedInterfacesFailed:
        return "QueryD11AdvancedInterfacesFailed";
    case DxContextError::D3DDevicesNotInitialized:
        return "D3DDevicesNotInitialized";
    case DxContextError::CreateSharedFenceFailed:
        return "CreateSharedFenceFailed";
    case DxContextError::CreateSharedFenceHandleFailed:
        return "CreateSharedFenceHandleFailed";
    case DxContextError::SharedFenceHandleNull:
        return "SharedFenceHandleNull";
    case DxContextError::OpenSharedFenceFailed:
        return "OpenSharedFenceFailed";
    case DxContextError::NotInitializedForFenceSignal:
        return "NotInitializedForFenceSignal";
    case DxContextError::SignalSharedFenceFailed:
        return "SignalSharedFenceFailed";
    case DxContextError::NotInitializedForD11FenceSignal:
        return "NotInitializedForD11FenceSignal";
    case DxContextError::SignalSharedFenceFromD11Failed:
        return "SignalSharedFenceFromD11Failed";
    case DxContextError::NotInitializedForD11FenceWait:
        return "NotInitializedForD11FenceWait";
    case DxContextError::WaitSharedFenceFromD11Failed:
        return "WaitSharedFenceFromD11Failed";
    }

    return "UnknownDxContextError";
}

namespace {

auto enableD3D12DebugLayer() -> void {
#ifdef _DEBUG
    winrt::com_ptr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debugController.put())))) {
        debugController->EnableDebugLayer();
        NRX_INFO("D3D12 debug layer enabled.");
    } else {
        NRX_WARN("D3D12 debug layer is unavailable.");
    }
#endif
}

} // namespace

class DxContext::Impl {
  public:
    Impl() = default;
    ~Impl() { releaseResources(); }

    Impl(const Impl&) = delete;
    auto operator=(const Impl&) -> Impl& = delete;
    Impl(Impl&&) = delete;
    auto operator=(Impl&&) -> Impl& = delete;

    auto init() -> std::expected<void, DxContextError> {
        releaseResources();

        if (const auto result = createFactory(); !result) {
            return result;
        }
        if (const auto result = createD12Device(); !result) {
            return result;
        }
        if (const auto result = createD12Queue(); !result) {
            return result;
        }
        if (const auto result = createD11Device(); !result) {
            return result;
        }
        if (const auto result = createSharedFence(); !result) {
            return result;
        }

        deviceLost.store(false);
        NRX_INFO("DxContext initialized successfully.");
        return {};
    }

    auto handleDeviceLost() -> std::expected<void, DxContextError> {
        NRX_WARN("Device lost detected. Recreating DX11/DX12 resources.");

        releaseResources();
        if (const auto result = init(); !result) {
            return result;
        }

        deviceLost.store(false);
        return {};
    }

    [[nodiscard]] auto getD11Device() const -> ID3D11Device5* { return d11Device.get(); }

    [[nodiscard]] auto getD11Context() const -> ID3D11DeviceContext4* { return d11Context.get(); }

    [[nodiscard]] auto getD12Device() const -> ID3D12Device* { return d12Device.get(); }

    [[nodiscard]] auto getD12Queue() const -> ID3D12CommandQueue* { return d12Queue.get(); }

    [[nodiscard]] auto getD11SharedFence() const -> ID3D11Fence* { return d11SharedFence.get(); }

    [[nodiscard]] auto getSharedFence() const -> ID3D12Fence* { return sharedFence.get(); }

    [[nodiscard]] auto getSharedFenceHandle() const -> DxContext::SharedHandle {
        return reinterpret_cast<void*>(sharedFenceHandle);
    }

    [[nodiscard]] auto getFenceValue() const -> uint64_t { return fenceValue; }

    auto signalSharedFence() -> std::expected<void, DxContextError> {
        if (!d12Queue || !sharedFence) {
            return std::unexpected(DxContextError::NotInitializedForFenceSignal);
        }

        fenceValue += 1;
        const HRESULT hr = d12Queue->Signal(sharedFence.get(), fenceValue);
        NRX_DX_CHECK(hr, "Failed to signal shared fence",
                             DxContextError::SignalSharedFenceFailed);

        return {};
    }

    auto signalSharedFenceFromD11() -> std::expected<void, DxContextError> {
        if (!d11Context || !d11SharedFence) {
            return std::unexpected(DxContextError::NotInitializedForD11FenceSignal);
        }

        fenceValue += 1;
        const HRESULT hr = d11Context->Signal(d11SharedFence.get(), fenceValue);
        NRX_DX_CHECK(hr, "Failed to signal shared fence from D3D11",
                             DxContextError::SignalSharedFenceFromD11Failed);

        return {};
    }

    auto waitSharedFenceFromD11(uint64_t targetValue) -> std::expected<void, DxContextError> {
        if (!d11Context || !d11SharedFence) {
            return std::unexpected(DxContextError::NotInitializedForD11FenceWait);
        }

        const HRESULT hr = d11Context->Wait(d11SharedFence.get(), targetValue);
        NRX_DX_CHECK(hr, "Failed to wait shared fence from D3D11",
                             DxContextError::WaitSharedFenceFromD11Failed);

        return {};
    }

    [[nodiscard]] auto isDeviceLost() const -> bool { return deviceLost.load(); }

    void notifyDeviceLost() { deviceLost.store(true); }

  private:
    auto createFactory() -> std::expected<void, DxContextError> {
        UINT factoryFlags = 0;
#ifdef _DEBUG
        factoryFlags = DXGI_CREATE_FACTORY_DEBUG;
#endif

        const HRESULT hr = CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(factory.put()));
        NRX_DX_CHECK(hr, "Failed to create DXGI factory", DxContextError::CreateFactoryFailed);

        return {};
    }

    [[nodiscard]] auto findHardwareAdapter() const
        -> std::expected<winrt::com_ptr<IDXGIAdapter4>, DxContextError> {
        if (cachedAdapter) {
            return cachedAdapter;
        }

        if (!factory) {
            return std::unexpected(DxContextError::FactoryNotInitialized);
        }

        for (UINT adapterIndex = 0;; ++adapterIndex) {
            winrt::com_ptr<IDXGIAdapter1> adapter;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(
                adapterIndex, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(adapter.put()));

            if (hr == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            NRX_DX_CHECK(hr, "Failed while enumerating adapters",
                                 DxContextError::AdapterEnumerationFailed);

            DXGI_ADAPTER_DESC1 descriptor{};
            adapter->GetDesc1(&descriptor);

            if ((descriptor.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
                continue;
            }

            if (SUCCEEDED(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0,
                                            __uuidof(ID3D12Device), nullptr))) {
                cachedAdapter = adapter.as<IDXGIAdapter4>();
                return cachedAdapter;
            }
        }

        return std::unexpected(DxContextError::NoHardwareAdapter);
    }

    auto createD12Device() -> std::expected<void, DxContextError> {
        enableD3D12DebugLayer();

        const auto adapterResult = findHardwareAdapter();
        if (!adapterResult) {
            return std::unexpected(adapterResult.error());
        }

        const auto& adapter = adapterResult.value();

        HRESULT hr =
            D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(d12Device.put()));
        if (FAILED(hr)) {
            hr = D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(d12Device.put()));
        }
        NRX_DX_CHECK(hr, "Failed to create D3D12 device",
                             DxContextError::CreateD12DeviceFailed);

        return {};
    }

    auto createD12Queue() -> std::expected<void, DxContextError> {
        if (!d12Device) {
            return std::unexpected(DxContextError::D12DeviceNotInitialized);
        }

        D3D12_COMMAND_QUEUE_DESC queueDescription{};
        queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queueDescription.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        queueDescription.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queueDescription.NodeMask = 0;

        const HRESULT hr =
            d12Device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(d12Queue.put()));
        NRX_DX_CHECK(hr, "Failed to create D3D12 command queue",
                             DxContextError::CreateD12QueueFailed);

        return {};
    }

    auto createD11Device() -> std::expected<void, DxContextError> {
        if (!factory) {
            return std::unexpected(DxContextError::FactoryNotInitialized);
        }

        const auto adapterResult = findHardwareAdapter();
        if (!adapterResult) {
            return std::unexpected(adapterResult.error());
        }

        const auto& adapter = adapterResult.value();

        UINT creationFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        creationFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        constexpr D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
        };

        winrt::com_ptr<ID3D11Device> baseDevice;
        winrt::com_ptr<ID3D11DeviceContext> baseContext;
        D3D_FEATURE_LEVEL createdFeatureLevel = D3D_FEATURE_LEVEL_11_0;

        const HRESULT hr = D3D11CreateDevice(
            adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, creationFlags, featureLevels,
            static_cast<UINT>(std::size(featureLevels)), D3D11_SDK_VERSION, baseDevice.put(),
            &createdFeatureLevel, baseContext.put());

        NRX_DX_CHECK(hr, "Failed to create D3D11 device", DxContextError::CreateD11DeviceFailed);

        d11Device = baseDevice.try_as<ID3D11Device5>();
        d11Context = baseContext.try_as<ID3D11DeviceContext4>();

        if (!d11Device || !d11Context) {
            NRX_ERROR("Failed to query D3D11 advanced interfaces (ID3D11Device5/ID3D11DeviceContext4).");
            return std::unexpected(DxContextError::QueryD11AdvancedInterfacesFailed);
        }

        NRX_INFO("D3D11 created with feature level: 0x{:X}",
                 static_cast<unsigned int>(createdFeatureLevel));

        return {};
    }

    auto createSharedFence() -> std::expected<void, DxContextError> {
        if (!d12Device || !d11Device) {
            return std::unexpected(DxContextError::D3DDevicesNotInitialized);
        }

        fenceValue = 0;

        const HRESULT fenceHr = d12Device->CreateFence(fenceValue, D3D12_FENCE_FLAG_SHARED,
                                                       IID_PPV_ARGS(sharedFence.put()));
        NRX_DX_CHECK(fenceHr, "Failed to create shared D3D12 fence",
                             DxContextError::CreateSharedFenceFailed);

        const HRESULT handleHr = d12Device->CreateSharedHandle(
            sharedFence.get(), nullptr, GENERIC_ALL, nullptr, &sharedFenceHandle);
        NRX_DX_CHECK(handleHr, "Failed to create shared fence handle",
                             DxContextError::CreateSharedFenceHandleFailed);

        if (sharedFenceHandle == nullptr) {
            return std::unexpected(DxContextError::SharedFenceHandleNull);
        }

        const HRESULT openFenceHr =
            d11Device->OpenSharedFence(sharedFenceHandle, IID_PPV_ARGS(d11SharedFence.put()));
        NRX_DX_CHECK(openFenceHr, "Failed to open shared fence on D3D11 device",
                             DxContextError::OpenSharedFenceFailed);

        return {};
    }

    void releaseResources() {
        d11SharedFence = nullptr;

        if (sharedFenceHandle != nullptr) {
            CloseHandle(sharedFenceHandle);
            sharedFenceHandle = nullptr;
        }

        sharedFence = nullptr;
        d11Context = nullptr;
        d11Device = nullptr;
        d12Queue = nullptr;
        d12Device = nullptr;
        cachedAdapter = nullptr;
        factory = nullptr;
        fenceValue = 0;
    }

    winrt::com_ptr<IDXGIFactory6> factory;

    winrt::com_ptr<ID3D11Device5> d11Device;
    winrt::com_ptr<ID3D11DeviceContext4> d11Context;

    winrt::com_ptr<ID3D12Device> d12Device;
    winrt::com_ptr<ID3D12CommandQueue> d12Queue;
    mutable winrt::com_ptr<IDXGIAdapter4> cachedAdapter;

    winrt::com_ptr<ID3D11Fence> d11SharedFence;
    winrt::com_ptr<ID3D12Fence> sharedFence;
    HANDLE sharedFenceHandle{nullptr};
    uint64_t fenceValue{0};
    std::atomic_bool deviceLost{false};
};

DxContext::DxContext() : impl(std::make_unique<Impl>()) {}

DxContext::~DxContext() = default;

auto DxContext::init() -> std::expected<void, DxContextError> { return impl->init(); }

auto DxContext::handleDeviceLost() -> std::expected<void, DxContextError> {
    return impl->handleDeviceLost();
}

auto DxContext::getD11Device() const -> ID3D11Device5* { return impl->getD11Device(); }

auto DxContext::getD11Context() const -> ID3D11DeviceContext4* { return impl->getD11Context(); }

auto DxContext::getD12Device() const -> ID3D12Device* { return impl->getD12Device(); }

auto DxContext::getD12Queue() const -> ID3D12CommandQueue* { return impl->getD12Queue(); }

auto DxContext::getD11SharedFence() const -> ID3D11Fence* { return impl->getD11SharedFence(); }

auto DxContext::getSharedFence() const -> ID3D12Fence* { return impl->getSharedFence(); }

auto DxContext::getSharedFenceHandle() const -> DxContext::SharedHandle {
    return impl->getSharedFenceHandle();
}

auto DxContext::getFenceValue() const -> uint64_t { return impl->getFenceValue(); }

auto DxContext::signalSharedFence() -> std::expected<void, DxContextError> {
    return impl->signalSharedFence();
}

auto DxContext::signalSharedFenceFromD11() -> std::expected<void, DxContextError> {
    return impl->signalSharedFenceFromD11();
}

auto DxContext::waitSharedFenceFromD11(uint64_t targetValue)
    -> std::expected<void, DxContextError> {
    return impl->waitSharedFenceFromD11(targetValue);
}

auto DxContext::isDeviceLost() const -> bool { return impl->isDeviceLost(); }

void DxContext::notifyDeviceLost() { impl->notifyDeviceLost(); }

} // namespace nrx::gfx
