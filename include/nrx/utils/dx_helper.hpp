#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include <d3d12.h>
#include <winrt/base.h>

#include "nrx/utils/logger.hpp"

namespace nrx::utils {

#define NRX_DX_CHECK(hr, message, errorCode)                                                    \
    if (const HRESULT checkedHr = (hr); FAILED(checkedHr)) {                                    \
        NRX_ERROR("{}: {}", (message), ::nrx::utils::DxHelper::getErrorString(checkedHr));     \
        return std::unexpected((errorCode));                                                    \
    }

template <typename T>
using DxResult = std::expected<winrt::com_ptr<T>, HRESULT>;

struct DxHelper {
    static auto getErrorString(HRESULT hr) -> std::string;
    static auto createBuffer(ID3D12Device* d12Device, std::uint64_t sizeBytes,
                             D3D12_RESOURCE_FLAGS flags, D3D12_HEAP_TYPE heapType)
        -> DxResult<ID3D12Resource>;
    static auto createUavBuffer(ID3D12Device* d12Device, std::uint64_t sizeBytes)
        -> DxResult<ID3D12Resource>;
    static auto createReadbackBuffer(ID3D12Device* d12Device, std::uint64_t sizeBytes)
        -> DxResult<ID3D12Resource>;
};

} // namespace nrx::utils
