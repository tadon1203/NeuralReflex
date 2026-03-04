#pragma once

#include <filesystem>
#include <mutex>
#include <optional>

#include "nrx/core/config.hpp"

namespace nrx::core {

class ConfigManager {
  public:
    explicit ConfigManager(std::filesystem::path path);

    bool reloadIfChanged();
    [[nodiscard]] auto getValidatedConfig() const -> AppConfig;

  private:
    auto ensureFileExistsWithDefault() -> bool;
    auto ensureModelsDirectoryExists() const -> bool;
    auto loadFromFile() -> std::optional<AppConfig>;
    auto validateAndResolveConfig(const AppConfig& config) const -> std::optional<AppConfig>;
    [[nodiscard]] auto resolveConfiguredModelPath(const std::filesystem::path& configuredPath) const
        -> std::filesystem::path;

    std::filesystem::path configPath;
    std::filesystem::file_time_type lastWriteTime;
    bool hasWriteTime{false};
    AppConfig currentConfig{defaultAppConfig()};
    mutable std::mutex mutex;
};

} // namespace nrx::core
