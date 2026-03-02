#pragma once

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include <spdlog/spdlog.h>

namespace nrx::utils {

class Logger {
  public:
    static void init();
};

} // namespace nrx::utils

#define NRX_TRACE(...)    SPDLOG_TRACE(__VA_ARGS__)
#define NRX_DEBUG(...)    SPDLOG_DEBUG(__VA_ARGS__)
#define NRX_INFO(...)     SPDLOG_INFO(__VA_ARGS__)
#define NRX_WARN(...)     SPDLOG_WARN(__VA_ARGS__)
#define NRX_ERROR(...)    SPDLOG_ERROR(__VA_ARGS__)
#define NRX_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
