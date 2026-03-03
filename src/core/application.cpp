#include "nrx/core/application.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>

#include <d3d12.h>

#include "core/platform.hpp"
#include "nrx/gfx/dx_context.hpp"
#include "nrx/gfx/gfx_bridge.hpp"
#include "nrx/gfx/screen_capturer.hpp"
#include "nrx/inference/inference_engine.hpp"
#include "nrx/inference/types.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::core {

namespace {

[[nodiscard]] auto resolveModelPath() -> std::filesystem::path {
    constexpr const char* kModelFileName = "ApxV7Harlan.onnx";
    const auto modelRelativePath = std::filesystem::path{"assets"} / "models" / kModelFileName;
    const std::array<std::filesystem::path, 3> candidates{
        modelRelativePath,
        std::filesystem::current_path() / modelRelativePath,
        std::filesystem::path{NRX_SOURCE_DIR} / modelRelativePath,
    };

    for (const auto& candidatePath : candidates) {
        if (std::filesystem::exists(candidatePath)) {
            return candidatePath;
        }
    }

    return candidates.back();
}

} // namespace

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

    gfxBridge = std::make_unique<nrx::gfx::GfxBridge>(dxContext.get());
    inferenceEngine = std::make_unique<nrx::inference::InferenceEngine>();

    const auto modelPath = resolveModelPath();
    if (!std::filesystem::exists(modelPath)) {
        NRX_CRITICAL("Inference model not found: {}", modelPath.string());
        shutdown();
        return -1;
    }
    if (const auto result = inferenceEngine->init(dxContext.get(), modelPath); !result) {
        NRX_CRITICAL("Failed to initialize InferenceEngine: {}",
                     nrx::inference::inferenceErrorToString(result.error()));
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

    const std::unique_lock<std::shared_mutex> lock(runtimeMutex);

    if (screenCapturer != nullptr) {
        screenCapturer->stop();
        screenCapturer.reset();
    }

    if (inferenceEngine != nullptr) {
        inferenceEngine->reset();
        inferenceEngine.reset();
    }

    if (gfxBridge != nullptr) {
        gfxBridge->reset();
        gfxBridge.reset();
    }

    dxContext.reset();
}

void Application::inferenceLoop(const std::stop_token& stopToken) {
    NRX_INFO("Inference thread started.");

    int emptyFrameStreak = 0;

    while (!stopToken.stop_requested()) {
        bool shouldYield = false;
        bool shouldSleep = false;

        {
            const std::shared_lock<std::shared_mutex> lock(runtimeMutex);
            if (dxContext == nullptr || screenCapturer == nullptr || gfxBridge == nullptr ||
                inferenceEngine == nullptr) {
                NRX_CRITICAL("Inference loop started without required runtime components.");
                return;
            }

            if (dxContext->checkDeviceLost()) {
                shouldSleep = true;
            } else {
                const auto frameResult = screenCapturer->acquireNextFrame();
                if (!frameResult) {
                    if (frameResult.error() == nrx::gfx::CaptureError::NoNewFrameAvailable) {
                        ++emptyFrameStreak;
                        shouldYield = true;
                        shouldSleep = ((emptyFrameStreak % 256) == 0);
                    } else {
                        NRX_WARN("Failed to acquire frame: {}",
                                 nrx::gfx::captureErrorToString(frameResult.error()));
                        shouldSleep = true;
                    }
                } else {
                    emptyFrameStreak = 0;

                    const auto mapResult = gfxBridge->registerTexture(frameResult.value());
                    if (!mapResult) {
                        NRX_WARN("Failed to register texture in GfxBridge: {}",
                                 nrx::gfx::bridgeErrorToString(mapResult.error()));
                        shouldSleep = true;
                    } else if (const auto syncResult = gfxBridge->synchronize(); !syncResult) {
                        NRX_WARN("Failed to synchronize GfxBridge: {}",
                                 nrx::gfx::bridgeErrorToString(syncResult.error()));
                        shouldSleep = true;
                    } else {
                        const auto inferenceResult = inferenceEngine->execute(
                            mapResult.value(), D3D12_RESOURCE_STATE_COMMON);
                        if (!inferenceResult) {
                            NRX_WARN("Inference execution failed: {}",
                                     nrx::inference::inferenceErrorToString(inferenceResult.error()));
                            shouldSleep = true;
                        }
                    }
                }
            }
        }

        if (shouldYield) {
            std::this_thread::yield();
        }
        if (shouldSleep) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
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
    const std::unique_lock<std::shared_mutex> lock(runtimeMutex);

    if (screenCapturer == nullptr || gfxBridge == nullptr || inferenceEngine == nullptr ||
        dxContext == nullptr) {
        NRX_CRITICAL("Runtime components are unavailable during device reset recovery.");
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

    gfxBridge->reset();
    inferenceEngine->reset();

    const auto modelPath = resolveModelPath();
    if (!std::filesystem::exists(modelPath)) {
        NRX_CRITICAL("Inference model not found during reset recovery: {}", modelPath.string());
        running.store(false);
        return;
    }
    if (const auto inferenceInitResult = inferenceEngine->init(dxContext.get(), modelPath);
        !inferenceInitResult) {
        NRX_CRITICAL("Failed to reinitialize InferenceEngine: {}",
                     nrx::inference::inferenceErrorToString(inferenceInitResult.error()));
        running.store(false);
        return;
    }

    NRX_INFO("ScreenCapturer, GfxBridge, and InferenceEngine reinitialized successfully.");
}
} // namespace nrx::core
