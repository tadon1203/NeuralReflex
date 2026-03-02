#pragma once

#include <atomic>
#include <memory>
#include <stop_token>
#include <thread>

namespace nrx::gfx {
class DxContext;
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
    void inferenceLoop(const std::stop_token& stopToken, nrx::gfx::DxContext* dxCtx);
    void overlayLoop();
    void reinitializeEnginesAfterDeviceReset();

    std::atomic_bool isRunning{false};
    std::unique_ptr<nrx::gfx::DxContext> dxContext;
    std::jthread inferenceThread;
};
} // namespace nrx::core
