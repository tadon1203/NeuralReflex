#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include "nrx/utils/logger.hpp"

namespace nrx::utils {

#define NRX_DX_CHECK(hr, message, errorCode)                                                    \
    if (const std::int32_t checkedHr = static_cast<std::int32_t>(hr); checkedHr < 0) {        \
        NRX_ERROR("{}: {}", (message), ::nrx::utils::DxHelper::getErrorString(checkedHr));     \
        return std::unexpected((errorCode));                                                    \
    }

struct DxHelper {
    static auto getErrorString(std::int32_t hr) -> std::string;
};

} // namespace nrx::utils
