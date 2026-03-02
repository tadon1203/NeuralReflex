#include "nrx/utils/logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>

namespace nrx::utils {

void Logger::init() {
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    consoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");

    auto logger = std::make_shared<spdlog::logger>("NRX", consoleSink);
    logger->set_level(spdlog::level::trace);

    spdlog::set_default_logger(logger);
}

} // namespace nrx::utils
