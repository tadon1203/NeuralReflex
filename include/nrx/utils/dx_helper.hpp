#pragma once

#include <string>

#include <Windows.h>
#include <winrt/base.h>

#include "nrx/utils/logger.hpp"

namespace nrx::utils {

#define NRX_DX_CHECK(hr, message, errorCode)                                                    \
    if (const HRESULT checkedHr = (hr); FAILED(checkedHr)) {                                   \
        NRX_ERROR("{}: {}", (message), ::nrx::utils::DxHelper::getErrorString(checkedHr));     \
        return std::unexpected((errorCode));                                                    \
    }

struct DxHelper {
    static auto getErrorString(HRESULT hr) -> std::string;
};

} // namespace nrx::utils
