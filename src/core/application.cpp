#include "nrx/core/application.hpp"

#include <memory>
#include <thread>

#include "nrx/gfx/dx_context.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::core {

Application::Application() { NRX_INFO("Initializing Application..."); }

Application::~Application() {
    shutdown();
    NRX_INFO("Shutting down...");
}

int Application::run() {
    NRX_INFO("Starting main loop...");
    isRunning.store(true);
    dxContext = std::make_unique<nrx::gfx::DxContext>();
    if (const auto result = dxContext->init(); !result) {
        NRX_CRITICAL("Failed to initialize DxContext: {}", result.error());
        shutdown();
        return -1;
    }

    nrx::gfx::DxContext* inferenceDxCtx = dxContext.get();
    inferenceThread = std::jthread(
        [this, inferenceDxCtx](const std::stop_token& st) { inferenceLoop(st, inferenceDxCtx); });

    overlayLoop();
    shutdown();

    NRX_INFO("Main loop exited.");

    return 0;
}

void Application::shutdown() {
    isRunning.store(false);

    if (inferenceThread.joinable()) {
        inferenceThread.request_stop();
        inferenceThread.join();
    }

    dxContext.reset();
}

void Application::inferenceLoop(const std::stop_token& stopToken, nrx::gfx::DxContext* dxCtx) {
    NRX_INFO("Inference thread started.");
    if (dxCtx == nullptr) {
        NRX_CRITICAL("Inference loop started without a valid DxContext.");
        return;
    }

    while (!stopToken.stop_requested()) {
        if (dxCtx->isDeviceLost()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void Application::overlayLoop() {
    NRX_INFO("Overlay thread started.");

    while (isRunning.load()) {
        if (dxContext != nullptr && dxContext->isDeviceLost()) {
            NRX_WARN("Overlay detected device lost. Reinitializing DxContext...");
            if (const auto result = dxContext->handleDeviceLost(); !result.has_value()) {
                NRX_CRITICAL("Failed to recover from device lost: {}", result.error());
                isRunning.store(false);
                break;
            }

            reinitializeEnginesAfterDeviceReset();
        }

        // Placeholder for overlay/UI rendering tick
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Application::reinitializeEnginesAfterDeviceReset() {
    NRX_INFO("Device reset completed. Engine reinitialization hook called.");
}
} // namespace nrx::core
