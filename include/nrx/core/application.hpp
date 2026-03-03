#pragma once

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <stop_token>
#include <thread>

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
    void inferenceLoop(const std::stop_token& stopToken);
    void overlayLoop();
    void reinitializeEnginesAfterDeviceReset();

    std::atomic_bool running{false};
    std::unique_ptr<nrx::gfx::DxContext> dxContext;
    std::unique_ptr<nrx::gfx::GfxBridge> gfxBridge;
    std::unique_ptr<nrx::gfx::ScreenCapturer> screenCapturer;
    std::unique_ptr<nrx::inference::InferenceEngine> inferenceEngine;
    mutable std::shared_mutex runtimeMutex;
    std::jthread inferenceThread;
};
} // namespace nrx::core
