#pragma once

#include <expected>
#include <string>

namespace nrx::core {

auto setupPlatformRuntime() -> std::expected<void, std::string>;

} // namespace nrx::core
