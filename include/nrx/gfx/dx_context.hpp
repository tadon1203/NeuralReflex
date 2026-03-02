#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <string>

struct ID3D11Device5;
struct ID3D11DeviceContext4;
struct ID3D11Fence;
struct ID3D12CommandQueue;
struct ID3D12Device;
struct ID3D12Fence;

namespace nrx::gfx {

class DxContext {
  public:
    using SharedHandle = void*;

    DxContext();
    ~DxContext();

    DxContext(const DxContext&) = delete;
    auto operator=(const DxContext&) -> DxContext& = delete;
    DxContext(DxContext&&) = delete;
    auto operator=(DxContext&&) -> DxContext& = delete;

    auto init() -> std::expected<void, std::string>;
    auto handleDeviceLost() -> std::expected<void, std::string>;

    [[nodiscard]] auto getD11Device() const -> ID3D11Device5*;
    [[nodiscard]] auto getD11Context() const -> ID3D11DeviceContext4*;
    [[nodiscard]] auto getD12Device() const -> ID3D12Device*;
    [[nodiscard]] auto getD12Queue() const -> ID3D12CommandQueue*;
    [[nodiscard]] auto getD11SharedFence() const -> ID3D11Fence*;
    [[nodiscard]] auto getSharedFence() const -> ID3D12Fence*;
    [[nodiscard]] auto getSharedFenceHandle() const -> SharedHandle;
    [[nodiscard]] auto getFenceValue() const -> uint64_t;

    auto signalSharedFence() -> std::expected<void, std::string>;
    auto signalSharedFenceFromD11() -> std::expected<void, std::string>;
    auto waitSharedFenceFromD11(uint64_t targetValue) -> std::expected<void, std::string>;
    [[nodiscard]] auto isDeviceLost() const -> bool;
    void notifyDeviceLost();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace nrx::gfx
