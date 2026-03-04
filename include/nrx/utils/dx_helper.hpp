#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include <d3d12.h>
#include <winrt/base.h>

#include "nrx/utils/logger.hpp"

namespace nrx::utils {

#define NRX_DX_CHECK(hr, message, errorCode)                                                    \
    if (const auto checkedHr = static_cast<std::int32_t>(hr); checkedHr < 0) {                 \
        NRX_ERROR("{}: {}", (message), ::nrx::utils::DxHelper::getErrorString(checkedHr));     \
        return std::unexpected((errorCode));                                                    \
    }

struct DxHelper {
    static auto getErrorString(std::int32_t hr) -> std::string;
    static auto createBuffer(ID3D12Device* d12Device, std::uint64_t sizeBytes,
                             D3D12_RESOURCE_FLAGS flags, D3D12_HEAP_TYPE heapType)
        -> std::expected<winrt::com_ptr<ID3D12Resource>, std::int32_t>;
};

} // namespace nrx::utils
