#pragma once

#include <cstdint>
#include <filesystem>

namespace nrx::core {

struct AppConfig {
    std::filesystem::path modelPath;
    float confidenceThreshold{0.45F};
    std::int32_t displayIndex{0};
};

[[nodiscard]] inline auto defaultAppConfig() -> AppConfig {
    return AppConfig{
        .modelPath = "model.onnx",
        .confidenceThreshold = 0.45F,
        .displayIndex = 0,
    };
}

[[nodiscard]] inline auto appConfigEquals(const AppConfig& lhs, const AppConfig& rhs) -> bool {
    return lhs.modelPath == rhs.modelPath && lhs.confidenceThreshold == rhs.confidenceThreshold &&
           lhs.displayIndex == rhs.displayIndex;
}

} // namespace nrx::core
