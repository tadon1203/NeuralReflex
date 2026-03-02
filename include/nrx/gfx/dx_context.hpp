#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string_view>

struct ID3D11Device5;
struct ID3D11DeviceContext4;
struct ID3D11Fence;
struct ID3D12CommandQueue;
struct ID3D12Device;
struct ID3D12Fence;

namespace nrx::gfx {

enum class DxContextError : std::uint8_t {
    CreateFactoryFailed,
    FactoryNotInitialized,
    AdapterEnumerationFailed,
    NoHardwareAdapter,
    CreateD12DeviceFailed,
    D12DeviceNotInitialized,
    CreateD12QueueFailed,
    CreateD11DeviceFailed,
    QueryD11AdvancedInterfacesFailed,
    D3DDevicesNotInitialized,
    CreateSharedFenceFailed,
    CreateSharedFenceHandleFailed,
    SharedFenceHandleNull,
    OpenSharedFenceFailed,
    NotInitializedForFenceSignal,
    SignalSharedFenceFailed,
    NotInitializedForD11FenceSignal,
    SignalSharedFenceFromD11Failed,
    NotInitializedForD11FenceWait,
    WaitSharedFenceFromD11Failed,
};

[[nodiscard]] auto dxContextErrorToString(DxContextError error) -> std::string_view;

class DxContext {
  public:
    using SharedHandle = void*;

    DxContext();
    ~DxContext();

    DxContext(const DxContext&) = delete;
    auto operator=(const DxContext&) -> DxContext& = delete;
    DxContext(DxContext&&) = delete;
    auto operator=(DxContext&&) -> DxContext& = delete;

    auto init() -> std::expected<void, DxContextError>;
    auto handleDeviceLost() -> std::expected<void, DxContextError>;

    [[nodiscard]] auto getD11Device() const -> ID3D11Device5*;
    [[nodiscard]] auto getD11Context() const -> ID3D11DeviceContext4*;
    [[nodiscard]] auto getD12Device() const -> ID3D12Device*;
    [[nodiscard]] auto getD12Queue() const -> ID3D12CommandQueue*;
    [[nodiscard]] auto getD11SharedFence() const -> ID3D11Fence*;
    [[nodiscard]] auto getSharedFence() const -> ID3D12Fence*;
    [[nodiscard]] auto getSharedFenceHandle() const -> SharedHandle;
    [[nodiscard]] auto getFenceValue() const -> uint64_t;

    auto signalSharedFence() -> std::expected<void, DxContextError>;
    auto signalSharedFenceFromD11() -> std::expected<void, DxContextError>;
    auto waitSharedFenceFromD11(uint64_t targetValue) -> std::expected<void, DxContextError>;
    [[nodiscard]] auto isDeviceLost() const -> bool;
    void notifyDeviceLost();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::gfx
