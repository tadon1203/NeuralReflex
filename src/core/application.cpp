#include "nrx/core/application.hpp"

#include <chrono>
#include <memory>
#include <shared_mutex>
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

constexpr std::chrono::seconds kConfigPollInterval{1};
constexpr const char* kConfigPath = "./config.json";

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

    configManager = std::make_unique<ConfigManager>(kConfigPath);
    activeConfig = configManager->getValidatedConfig();

    running.store(true);
    dxContext = std::make_unique<nrx::gfx::DxContext>();
    if (const auto result = dxContext->init(); !result) {
        NRX_CRITICAL("Failed to initialize DxContext: {}",
                     nrx::gfx::dxContextErrorToString(result.error()));
        shutdown();
        return -1;
    }

    screenCapturer = std::make_unique<nrx::gfx::ScreenCapturer>(dxContext.get());
    if (const auto result = screenCapturer->init(activeConfig.displayIndex); !result) {
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

    if (const auto result = inferenceEngine->init(dxContext.get(), activeConfig.modelPath); !result) {
        NRX_CRITICAL("Failed to initialize InferenceEngine: {}",
                     nrx::inference::inferenceErrorToString(result.error()));
        shutdown();
        return -1;
    }

    inferenceEngine->setScoreThreshold(activeConfig.confidenceThreshold);

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
    configManager.reset();
}

void Application::inferenceLoop(const std::stop_token& stopToken) {
    NRX_INFO("Inference thread started.");

    int emptyFrameStreak = 0;
    auto lastConfigCheck = std::chrono::steady_clock::now();

    while (!stopToken.stop_requested()) {
        const auto now = std::chrono::steady_clock::now();
        if ((now - lastConfigCheck) >= kConfigPollInterval) {
            handleConfigUpdate();
            lastConfigCheck = now;
        }

        const auto frameResult = processSingleFrame();
        if (frameResult == FrameResult::NoFrame) {
            ++emptyFrameStreak;
            std::this_thread::yield();
            if ((emptyFrameStreak % 256) == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            continue;
        }

        emptyFrameStreak = 0;
        if (frameResult == FrameResult::Error || frameResult == FrameResult::DeviceLost) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

auto Application::processSingleFrame() -> FrameResult {
    const std::shared_lock<std::shared_mutex> lock(runtimeMutex);

    if (dxContext == nullptr || screenCapturer == nullptr || gfxBridge == nullptr ||
        inferenceEngine == nullptr) {
        NRX_CRITICAL("Inference loop started without required runtime components.");
        return FrameResult::Error;
    }

    if (dxContext->checkDeviceLost()) {
        return FrameResult::DeviceLost;
    }

    const auto frameResult = screenCapturer->acquireNextFrame();
    if (!frameResult) {
        if (frameResult.error() == nrx::gfx::CaptureError::NoNewFrameAvailable) {
            return FrameResult::NoFrame;
        }

        NRX_WARN("Failed to acquire frame: {}", nrx::gfx::captureErrorToString(frameResult.error()));
        return FrameResult::Error;
    }

    const auto mapResult = gfxBridge->registerTexture(frameResult.value());
    if (!mapResult) {
        NRX_WARN("Failed to register texture in GfxBridge: {}",
                 nrx::gfx::bridgeErrorToString(mapResult.error()));
        return FrameResult::Error;
    }

    if (const auto syncResult = gfxBridge->synchronize(); !syncResult) {
        NRX_WARN("Failed to synchronize GfxBridge: {}",
                 nrx::gfx::bridgeErrorToString(syncResult.error()));
        return FrameResult::Error;
    }

    const auto inferenceResult = inferenceEngine->execute(mapResult.value(), D3D12_RESOURCE_STATE_COMMON);
    if (!inferenceResult) {
        NRX_WARN("Inference execution failed: {}",
                 nrx::inference::inferenceErrorToString(inferenceResult.error()));
        return FrameResult::Error;
    }

    return FrameResult::Success;
}

void Application::handleConfigUpdate() {
    if (configManager == nullptr) {
        return;
    }

    if (configManager->reloadIfChanged()) {
        applyConfig(configManager->getValidatedConfig());
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

void Application::applyConfig(const AppConfig& config) {
    if (appConfigEquals(activeConfig, config)) {
        return;
    }

    const std::unique_lock<std::shared_mutex> lock(runtimeMutex);

    if (screenCapturer == nullptr || gfxBridge == nullptr || inferenceEngine == nullptr ||
        dxContext == nullptr) {
        NRX_WARN("Skipped config apply because runtime components are unavailable.");
        return;
    }

    AppConfig nextConfig = activeConfig;

    if (config.displayIndex != activeConfig.displayIndex) {
        if (screenCapturer->reconfigure(config.displayIndex)) {
            gfxBridge->reset();
            nextConfig.displayIndex = config.displayIndex;
        } else {
            NRX_WARN("Display change failed. Keeping display index {}.", activeConfig.displayIndex);
        }
    }

    if (config.modelPath != activeConfig.modelPath ||
        config.confidenceThreshold != activeConfig.confidenceThreshold) {
        const auto targetModelPath =
            (config.modelPath != activeConfig.modelPath) ? config.modelPath : activeConfig.modelPath;

        if (inferenceEngine->update(targetModelPath, config.confidenceThreshold)) {
            nextConfig.modelPath = targetModelPath;
            nextConfig.confidenceThreshold = config.confidenceThreshold;
        } else {
            NRX_WARN("Inference settings update failed. Keeping previous model/threshold.");
        }
    }

    if (appConfigEquals(nextConfig, activeConfig)) {
        return;
    }

    activeConfig = nextConfig;
    NRX_INFO("Config reloaded and applied.");
}

void Application::reinitializeEnginesAfterDeviceReset() {
    const std::unique_lock<std::shared_mutex> lock(runtimeMutex);

    if (screenCapturer == nullptr || gfxBridge == nullptr || inferenceEngine == nullptr ||
        dxContext == nullptr) {
        NRX_CRITICAL("Runtime components are unavailable during device reset recovery.");
        running.store(false);
        return;
    }

    if (!screenCapturer->reconfigure(activeConfig.displayIndex)) {
        NRX_CRITICAL("Failed to reconfigure ScreenCapturer during device reset recovery.");
        running.store(false);
        return;
    }

    gfxBridge->reset();

    if (!inferenceEngine->reinitialize()) {
        NRX_CRITICAL("Failed to reinitialize InferenceEngine during device reset recovery.");
        running.store(false);
        return;
    }

    NRX_INFO("ScreenCapturer, GfxBridge, and InferenceEngine reinitialized successfully.");
}

} // namespace nrx::core
