#pragma once

#include <string>

#include <Windows.h>
#include <winrt/base.h>

#include "nrx/utils/logger.hpp"

namespace nrx::utils {

#define NRX_DX_CHECK(hr, message)                                                                  \
    if (const HRESULT checkedHr = (hr); FAILED(checkedHr)) {                                       \
        const auto fullMsg =                                                                       \
            std::format("{}: {}", (message), ::nrx::utils::DxHelper::getErrorString(checkedHr));   \
        NRX_CRITICAL("{}", fullMsg);                                                               \
        return std::unexpected(fullMsg);                                                           \
    }

struct DxHelper {
    static auto getErrorString(HRESULT hr) -> std::string;
};

} // namespace nrx::utils
