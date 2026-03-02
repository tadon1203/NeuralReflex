#include "nrx/core/application.hpp"

#include <memory>
#include <thread>

#include "core/platform.hpp"
#include "nrx/gfx/dx_context.hpp"
#include "nrx/gfx/screen_capturer.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::core {

Application::Application() { NRX_INFO("Initializing Application..."); }

Application::~Application() {
    shutdown();
    NRX_INFO("Shutting down...");
}

int Application::run() {
    NRX_INFO("Starting main loop...");

    if (const auto platformResult = setupPlatformRuntime(); !platformResult) {
        NRX_CRITICAL("Failed to setup platform runtime: {}", platformResult.error());
        return -1;
    }

    running.store(true);
    dxContext = std::make_unique<nrx::gfx::DxContext>();
    if (const auto result = dxContext->init(); !result) {
        NRX_CRITICAL("Failed to initialize DxContext: {}",
                     nrx::gfx::dxContextErrorToString(result.error()));
        shutdown();
        return -1;
    }

    screenCapturer = std::make_unique<nrx::gfx::ScreenCapturer>(dxContext.get());
    if (const auto result = screenCapturer->init(); !result) {
        NRX_CRITICAL("Failed to initialize ScreenCapturer: {}",
                     nrx::gfx::captureErrorToString(result.error()));
        shutdown();
        return -1;
    }

    if (const auto result = screenCapturer->start(); !result) {
        NRX_CRITICAL("Failed to start ScreenCapturer: {}",
                     nrx::gfx::captureErrorToString(result.error()));
        shutdown();
        return -1;
    }

    inferenceThread = std::jthread([this](const std::stop_token& st) { inferenceLoop(st); });

    overlayLoop();
    shutdown();

    NRX_INFO("Main loop exited.");

    return 0;
}

void Application::shutdown() {
    running.store(false);

    if (inferenceThread.joinable()) {
        inferenceThread.request_stop();
        inferenceThread.join();
    }

    if (screenCapturer != nullptr) {
        screenCapturer->stop();
        screenCapturer.reset();
    }

    dxContext.reset();
}

void Application::inferenceLoop(const std::stop_token& stopToken) {
    NRX_INFO("Inference thread started.");
    if (dxContext == nullptr || screenCapturer == nullptr) {
        NRX_CRITICAL("Inference loop started without required runtime components.");
        return;
    }

    int emptyFrameStreak = 0;

    while (!stopToken.stop_requested()) {
        if (dxContext->checkDeviceLost()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        const auto frameResult = screenCapturer->acquireNextFrame();
        if (!frameResult) {
            if (frameResult.error() == nrx::gfx::CaptureError::NoNewFrameAvailable) {
                ++emptyFrameStreak;
                std::this_thread::yield();
                if ((emptyFrameStreak % 256) == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                continue;
            }

            NRX_WARN("Failed to acquire frame: {}",
                     nrx::gfx::captureErrorToString(frameResult.error()));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        emptyFrameStreak = 0;

        // TODO: Wire the acquired frame into the inference pipeline.
    }
}

void Application::overlayLoop() {
    NRX_INFO("Overlay thread started.");

    while (running.load()) {
        if (dxContext != nullptr && dxContext->checkDeviceLost()) {
            NRX_WARN("Overlay detected device lost. Reinitializing DxContext...");
            if (const auto result = dxContext->handleDeviceLost(); !result) {
                NRX_CRITICAL("Failed to recover from device lost: {}",
                             nrx::gfx::dxContextErrorToString(result.error()));
                running.store(false);
                break;
            }

            reinitializeEnginesAfterDeviceReset();
        }

        // Placeholder for overlay/UI rendering tick
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Application::reinitializeEnginesAfterDeviceReset() {
    if (screenCapturer == nullptr) {
        NRX_CRITICAL("ScreenCapturer is unavailable during device reset recovery.");
        running.store(false);
        return;
    }

    NRX_INFO("Reinitializing ScreenCapturer after device reset...");

    screenCapturer->stop();

    if (const auto result = screenCapturer->init(); !result) {
        NRX_CRITICAL("Failed to reinitialize ScreenCapturer: {}",
                     nrx::gfx::captureErrorToString(result.error()));
        running.store(false);
        return;
    }

    if (const auto result = screenCapturer->start(); !result) {
        NRX_CRITICAL("Failed to restart ScreenCapturer: {}",
                     nrx::gfx::captureErrorToString(result.error()));
        running.store(false);
        return;
    }

    NRX_INFO("ScreenCapturer reinitialized successfully.");
}
} // namespace nrx::core
