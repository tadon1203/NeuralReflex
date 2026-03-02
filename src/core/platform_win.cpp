#include <expected>
#include <format>
#include <string>

#include <objbase.h>
#include <winrt/base.h>

#include "core/platform.hpp"

namespace nrx::core {

auto setupPlatformRuntime() -> std::expected<void, std::string> {
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        return {};
    } catch (const winrt::hresult_error& e) {
        if (e.code() == RPC_E_CHANGED_MODE) {
            return {};
        }

        return std::unexpected(
            std::format("Failed to initialize WinRT apartment: {}", winrt::to_string(e.message())));
    }
}

} // namespace nrx::core
