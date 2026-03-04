#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <stop_token>
#include <thread>

#include "nrx/core/config.hpp"
#include "nrx/core/config_manager.hpp"

namespace nrx::gfx {
class DxContext;
class GfxBridge;
class ScreenCapturer;
}
namespace nrx::inference {
class InferenceEngine;
}

namespace nrx::core {

class Application {
  public:
    Application();
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;

    int run();
    void shutdown();

  private:
    enum class FrameResult : std::uint8_t {
        Success,
        NoFrame,
        Error,
        DeviceLost,
    };

    void inferenceLoop(const std::stop_token& stopToken);
    auto processSingleFrame() -> FrameResult;
    void handleConfigUpdate();
    void overlayLoop();
    void applyConfig(const AppConfig& config);
    void reinitializeEnginesAfterDeviceReset();

    std::atomic_bool running{false};
    AppConfig activeConfig{defaultAppConfig()};
    std::unique_ptr<ConfigManager> configManager;
    std::unique_ptr<nrx::gfx::DxContext> dxContext;
    std::unique_ptr<nrx::gfx::GfxBridge> gfxBridge;
    std::unique_ptr<nrx::gfx::ScreenCapturer> screenCapturer;
    std::unique_ptr<nrx::inference::InferenceEngine> inferenceEngine;
    mutable std::shared_mutex runtimeMutex;
    std::jthread inferenceThread;
};
} // namespace nrx::core
