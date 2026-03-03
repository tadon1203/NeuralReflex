#include "nrx/gfx/gfx_bridge.hpp"

#include <cstdint>
#include <expected>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <Windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <winrt/base.h>

#include "nrx/gfx/dx_context.hpp"
#include "nrx/utils/dx_helper.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::gfx {

auto bridgeErrorToString(BridgeError error) -> std::string_view {
    switch (error) {
    case BridgeError::InvalidArguments:
        return "InvalidArguments";
    case BridgeError::DxContextNull:
        return "DxContextNull";
    case BridgeError::ResourceSharingFailed:
        return "ResourceSharingFailed";
    case BridgeError::SyncFailed:
        return "SyncFailed";
    case BridgeError::DeviceLost:
        return "DeviceLost";
    }

    return "UnknownBridgeError";
}

class GfxBridge::Impl {
  public:
    explicit Impl(DxContext* context) : dxContext(context) {}

    ~Impl() { reset(); }

    Impl(const Impl&) = delete;
    auto operator=(const Impl&) -> Impl& = delete;
    Impl(Impl&&) = delete;
    auto operator=(Impl&&) -> Impl& = delete;

    auto registerTexture(ID3D11Texture2D* d11Texture)
        -> std::expected<ID3D12Resource*, BridgeError> {
        const std::lock_guard<std::mutex> lock(bridgeMutex);

        if (d11Texture == nullptr) {
            return std::unexpected(BridgeError::InvalidArguments);
        }
        if (dxContext == nullptr) {
            return std::unexpected(BridgeError::DxContextNull);
        }
        if (dxContext->checkDeviceLost()) {
            return std::unexpected(BridgeError::DeviceLost);
        }

        if (const auto found = resourcesByTexture.find(d11Texture);
            found != resourcesByTexture.end()) {
            currentTextureKey = d11Texture;
            return found->second.d12Resource.get();
        }

        auto* d12Device = dxContext->getD12Device();
        if (d12Device == nullptr) {
            return std::unexpected(BridgeError::ResourceSharingFailed);
        }

        winrt::com_ptr<IDXGIResource1> dxgiResource;
        const HRESULT queryHr = d11Texture->QueryInterface(IID_PPV_ARGS(dxgiResource.put()));
        NRX_DX_CHECK(queryHr, "registerTexture: failed to query IDXGIResource1",
                     BridgeError::ResourceSharingFailed);

        HANDLE sharedHandle = nullptr;
        const HRESULT createHandleHr =
            dxgiResource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &sharedHandle);
        NRX_DX_CHECK(createHandleHr, "registerTexture: failed to create shared handle",
                     BridgeError::ResourceSharingFailed);

        if (sharedHandle == nullptr) {
            NRX_ERROR("registerTexture: CreateSharedHandle returned null handle");
            return std::unexpected(BridgeError::ResourceSharingFailed);
        }

        winrt::com_ptr<ID3D12Resource> d12Resource;
        const HRESULT openHandleHr =
            d12Device->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(d12Resource.put()));
        if (FAILED(openHandleHr)) {
            CloseHandle(sharedHandle);
            NRX_DX_CHECK(openHandleHr, "registerTexture: failed to open shared handle on D3D12",
                         BridgeError::ResourceSharingFailed);
        }

        ResourceEntry entry;
        entry.d11Texture.copy_from(d11Texture);
        entry.d12Resource = std::move(d12Resource);
        entry.sharedHandle = sharedHandle;

        const auto [insertedIt, inserted] =
            resourcesByTexture.emplace(d11Texture, std::move(entry));
        static_cast<void>(inserted);
        currentTextureKey = d11Texture;

        return insertedIt->second.d12Resource.get();
    }

    auto synchronize() -> std::expected<void, BridgeError> {
        const std::lock_guard<std::mutex> lock(bridgeMutex);

        if (currentTextureKey == nullptr) {
            return std::unexpected(BridgeError::InvalidArguments);
        }
        if (dxContext == nullptr) {
            return std::unexpected(BridgeError::DxContextNull);
        }
        if (dxContext->checkDeviceLost()) {
            return std::unexpected(BridgeError::DeviceLost);
        }

        auto* d11Context = dxContext->getD11Context();
        auto* d12Queue = dxContext->getD12Queue();
        auto* sharedFence = dxContext->getSharedFence();

        if (d11Context == nullptr || d12Queue == nullptr || sharedFence == nullptr) {
            return std::unexpected(BridgeError::SyncFailed);
        }

        d11Context->Flush();

        if (const auto signalResult = dxContext->signalSharedFenceFromD11(); !signalResult) {
            NRX_ERROR("synchronize: signalSharedFenceFromD11 failed: {}",
                      dxContextErrorToString(signalResult.error()));
            return std::unexpected(BridgeError::SyncFailed);
        }

        const uint64_t fenceValue = dxContext->getFenceValue();
        const HRESULT waitHr = d12Queue->Wait(sharedFence, fenceValue);
        NRX_DX_CHECK(waitHr, "synchronize: D3D12 queue wait failed", BridgeError::SyncFailed);

        return {};
    }

    void reset() {
        const std::lock_guard<std::mutex> lock(bridgeMutex);

        for (auto& [key, entry] : resourcesByTexture) {
            static_cast<void>(key);
            if (entry.sharedHandle != nullptr) {
                CloseHandle(entry.sharedHandle);
                entry.sharedHandle = nullptr;
            }
        }

        resourcesByTexture.clear();
        currentTextureKey = nullptr;
    }

  private:
    struct ResourceEntry {
        winrt::com_ptr<ID3D11Texture2D> d11Texture;
        winrt::com_ptr<ID3D12Resource> d12Resource;
        HANDLE sharedHandle{nullptr};
    };

    DxContext* dxContext{nullptr};
    std::mutex bridgeMutex;
    std::unordered_map<ID3D11Texture2D*, ResourceEntry> resourcesByTexture;
    ID3D11Texture2D* currentTextureKey{nullptr};
};

GfxBridge::GfxBridge(DxContext* context) : impl(std::make_unique<Impl>(context)) {}

GfxBridge::~GfxBridge() = default;

auto GfxBridge::registerTexture(ID3D11Texture2D* d11Texture)
    -> std::expected<ID3D12Resource*, BridgeError> {
    return impl->registerTexture(d11Texture);
}

auto GfxBridge::synchronize() -> std::expected<void, BridgeError> { return impl->synchronize(); }

void GfxBridge::reset() { impl->reset(); }

} // namespace nrx::gfx
