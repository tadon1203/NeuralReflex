#include "nrx/core/config_manager.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <optional>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

#include "nrx/utils/logger.hpp"

namespace nrx::core {

namespace {

[[nodiscard]] auto appConfigToJson(const AppConfig& config) -> nlohmann::json {
    return nlohmann::json{
        {"model_path", config.modelPath.string()},
        {"confidence_threshold", config.confidenceThreshold},
        {"display_index", config.displayIndex},
    };
}

} // namespace

ConfigManager::ConfigManager(std::filesystem::path path) : configPath(std::move(path)) {
    if (!ensureModelsDirectoryExists()) {
        NRX_WARN("Failed to prepare models directory. Falling back to in-memory defaults.");
    }

    if (!ensureFileExistsWithDefault()) {
        NRX_WARN("Failed to prepare config file. Falling back to in-memory defaults.");
        if (const auto validatedDefault = validateAndResolveConfig(defaultAppConfig());
            validatedDefault.has_value()) {
            currentConfig = validatedDefault.value();
        }
        return;
    }

    if (const auto loaded = loadFromFile(); loaded.has_value()) {
        currentConfig = loaded.value();
    } else {
        NRX_WARN("Failed to load config from '{}'. Using defaults.", configPath.string());
        if (const auto validatedDefault = validateAndResolveConfig(defaultAppConfig());
            validatedDefault.has_value()) {
            currentConfig = validatedDefault.value();
        }
    }

    std::error_code errorCode;
    const auto writeTime = std::filesystem::last_write_time(configPath, errorCode);
    if (errorCode) {
        NRX_WARN("Failed to query config write time '{}': {}", configPath.string(),
                 errorCode.message());
        return;
    }

    lastWriteTime = writeTime;
    hasWriteTime = true;
}

bool ConfigManager::reloadIfChanged() {
    std::error_code errorCode;
    if (!std::filesystem::exists(configPath, errorCode)) {
        if (errorCode) {
            NRX_WARN("Failed to check config file existence '{}': {}", configPath.string(),
                     errorCode.message());
        }
        return false;
    }

    const auto writeTime = std::filesystem::last_write_time(configPath, errorCode);
    if (errorCode) {
        NRX_WARN("Failed to query config write time '{}': {}", configPath.string(),
                 errorCode.message());
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex);
    if (hasWriteTime && writeTime == lastWriteTime) {
        return false;
    }

    const auto loadedConfig = loadFromFile();
    if (!loadedConfig.has_value()) {
        return false;
    }

    currentConfig = loadedConfig.value();
    lastWriteTime = writeTime;
    hasWriteTime = true;
    return true;
}

auto ConfigManager::getValidatedConfig() const -> AppConfig {
    const std::lock_guard<std::mutex> lock(mutex);
    return currentConfig;
}

auto ConfigManager::ensureFileExistsWithDefault() -> bool {
    std::error_code errorCode;
    if (std::filesystem::exists(configPath, errorCode)) {
        return !errorCode;
    }
    if (errorCode) {
        NRX_ERROR("Failed to check config file existence '{}': {}", configPath.string(),
                  errorCode.message());
        return false;
    }

    const auto parentPath = configPath.parent_path();
    if (!parentPath.empty()) {
        std::filesystem::create_directories(parentPath, errorCode);
        if (errorCode) {
            NRX_ERROR("Failed to create config directory '{}': {}", parentPath.string(),
                      errorCode.message());
            return false;
        }
    }

    std::ofstream output(configPath);
    if (!output.is_open()) {
        NRX_ERROR("Failed to create default config file '{}'.", configPath.string());
        return false;
    }

    output << std::setw(4) << appConfigToJson(defaultAppConfig()) << '\n';
    if (!output.good()) {
        NRX_ERROR("Failed to write default config file '{}'.", configPath.string());
        return false;
    }

    NRX_INFO("Created default config at '{}'.", configPath.string());
    return true;
}

auto ConfigManager::ensureModelsDirectoryExists() const -> bool {
    const auto modelsRoot = std::filesystem::current_path() / "models";

    std::error_code errorCode;
    if (std::filesystem::exists(modelsRoot, errorCode)) {
        if (errorCode) {
            NRX_ERROR("Failed to check models directory '{}': {}", modelsRoot.string(),
                      errorCode.message());
            return false;
        }
        return true;
    }

    std::filesystem::create_directories(modelsRoot, errorCode);
    if (errorCode) {
        NRX_ERROR("Failed to create models directory '{}': {}", modelsRoot.string(),
                  errorCode.message());
        return false;
    }

    NRX_INFO("Created models directory: {}", modelsRoot.string());
    return true;
}

auto ConfigManager::loadFromFile() -> std::optional<AppConfig> {
    std::ifstream input(configPath);
    if (!input.is_open()) {
        NRX_ERROR("Failed to open config file '{}'.", configPath.string());
        return std::nullopt;
    }

    nlohmann::json jsonValue;
    try {
        input >> jsonValue;
    } catch (const nlohmann::json::exception& error) {
        NRX_ERROR("Failed to parse config JSON '{}': {}", configPath.string(), error.what());
        return std::nullopt;
    }

    try {
        const AppConfig loadedConfig{
            .modelPath = std::filesystem::path{jsonValue.at("model_path").get<std::string>()},
            .confidenceThreshold = jsonValue.at("confidence_threshold").get<float>(),
            .displayIndex = jsonValue.at("display_index").get<std::int32_t>(),
        };

        return validateAndResolveConfig(loadedConfig);
    } catch (const nlohmann::json::exception& error) {
        NRX_ERROR("Config validation failed for '{}': {}", configPath.string(), error.what());
        return std::nullopt;
    }
}

auto ConfigManager::validateAndResolveConfig(const AppConfig& config) const
    -> std::optional<AppConfig> {
    if (config.modelPath.empty()) {
        NRX_ERROR("Config validation failed: 'model_path' must not be empty.");
        return std::nullopt;
    }

    if (!std::isfinite(config.confidenceThreshold) || config.confidenceThreshold < 0.0F ||
        config.confidenceThreshold > 1.0F) {
        NRX_ERROR("Config validation failed: 'confidence_threshold' must be in [0.0, 1.0].");
        return std::nullopt;
    }

    if (config.displayIndex < 0) {
        NRX_ERROR("Config validation failed: 'display_index' must be >= 0.");
        return std::nullopt;
    }

    if (!ensureModelsDirectoryExists()) {
        return std::nullopt;
    }

    return AppConfig{
        .modelPath = resolveConfiguredModelPath(config.modelPath),
        .confidenceThreshold = config.confidenceThreshold,
        .displayIndex = config.displayIndex,
    };
}

auto ConfigManager::resolveConfiguredModelPath(const std::filesystem::path& configuredPath) const
    -> std::filesystem::path {
    const auto modelsRoot = std::filesystem::current_path() / "models";
    const auto modelRelativePath =
        configuredPath.is_absolute() ? configuredPath.filename() : configuredPath;
    return (modelsRoot / modelRelativePath).lexically_normal();
}

} // namespace nrx::core
