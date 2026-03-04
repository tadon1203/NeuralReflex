#include "nrx/inference/inference_engine.hpp"

#include <expected>
#include <filesystem>
#include <memory>

#include "nrx/gfx/dx_context.hpp"
#include "nrx/inference/image_preprocessor.hpp"
#include "nrx/inference/ort_session_manager.hpp"
#include "nrx/inference/postprocessor.hpp"
#include "nrx/inference/resource_transition.hpp"
#include "nrx/inference/types.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::inference {

class InferenceEngine::Impl {
  public:
    auto init(nrx::gfx::DxContext* context, const std::filesystem::path& modelPath)
        -> std::expected<void, InferenceError> {
        resetRuntime();

        if (context == nullptr || modelPath.empty()) {
            return std::unexpected(InferenceError::InvalidArguments);
        }
        if (context->checkDeviceLost()) {
            return std::unexpected(InferenceError::DeviceLost);
        }

        if (const auto preprocessResult = preprocessor.init(context); !preprocessResult) {
            resetRuntime();
            return std::unexpected(preprocessResult.error());
        }

        if (const auto sessionResult = sessionManager.init(context, modelPath); !sessionResult) {
            resetRuntime();
            return std::unexpected(sessionResult.error());
        }

        if (const auto postprocessorResult = postprocessor.init(
                context, sessionManager.outputShape(), sessionManager.getInputResolution());
            !postprocessorResult) {
            resetRuntime();
            return std::unexpected(postprocessorResult.error());
        }

        postprocessor.setScoreThreshold(activeScoreThreshold);

        dxContext = context;
        activeModelPath = modelPath;
        initialized = true;
        return {};
    }

    auto execute(ID3D12Resource* inputTexture, D3D12_RESOURCE_STATES currentState)
        -> std::expected<DetectionResults, InferenceError> {
        if (!initialized) {
            return std::unexpected(InferenceError::NotInitialized);
        }
        if (inputTexture == nullptr) {
            return std::unexpected(InferenceError::InvalidArguments);
        }

        const ResourceTransition transition{
            .fromState = currentState,
            .toState = currentState,
        };
        const auto preprocessed = preprocessor.preprocess(inputTexture, transition);
        if (!preprocessed) {
            return std::unexpected(preprocessed.error());
        }

        const auto sessionOutput = sessionManager.run(preprocessed.value().resource);
        if (!sessionOutput) {
            return std::unexpected(sessionOutput.error());
        }

        const auto postDispatchResult = postprocessor.dispatch(sessionOutput.value().resource,
                                                               sessionOutput.value().currentState);
        if (!postDispatchResult) {
            return std::unexpected(postDispatchResult.error());
        }

        return postprocessor.readbackFinalResults();
    }

    auto update(const std::filesystem::path& modelPath, float confidenceThreshold) -> bool {
        if (modelPath.empty()) {
            return false;
        }

        auto* const previousContext = dxContext;
        const auto previousModelPath = activeModelPath;
        const auto previousThreshold = activeScoreThreshold;
        const bool previousInitialized = initialized;

        activeScoreThreshold = confidenceThreshold;

        if (!previousInitialized) {
            activeScoreThreshold = previousThreshold;
            return false;
        }

        if (modelPath == previousModelPath) {
            postprocessor.setScoreThreshold(confidenceThreshold);
            return true;
        }

        if (previousContext == nullptr) {
            activeScoreThreshold = previousThreshold;
            return false;
        }

        if (const auto initResult = init(previousContext, modelPath); !initResult) {
            NRX_WARN("InferenceEngine update failed for model '{}': {}", modelPath.string(),
                     inferenceErrorToString(initResult.error()));

            activeScoreThreshold = previousThreshold;
            if (const auto restoreResult = init(previousContext, previousModelPath);
                !restoreResult) {
                NRX_CRITICAL("InferenceEngine rollback failed for model '{}': {}",
                             previousModelPath.string(),
                             inferenceErrorToString(restoreResult.error()));
                return false;
            }
            postprocessor.setScoreThreshold(previousThreshold);
            return false;
        }

        return true;
    }

    auto reinitialize() -> bool {
        if (dxContext == nullptr || activeModelPath.empty()) {
            return false;
        }

        const auto preservedThreshold = activeScoreThreshold;
        if (const auto initResult = init(dxContext, activeModelPath); !initResult) {
            NRX_ERROR("InferenceEngine reinitialize failed: {}",
                      inferenceErrorToString(initResult.error()));
            return false;
        }

        activeScoreThreshold = preservedThreshold;
        postprocessor.setScoreThreshold(activeScoreThreshold);
        return true;
    }

    void setScoreThreshold(float value) {
        activeScoreThreshold = value;
        if (initialized) {
            postprocessor.setScoreThreshold(value);
        }
    }

    void reset() {
        resetRuntime();
        dxContext = nullptr;
        activeModelPath.clear();
        activeScoreThreshold = 0.45F;
    }

  private:
    void resetRuntime() {
        initialized = false;
        postprocessor.reset();
        sessionManager.reset();
        preprocessor.reset();
    }

    bool initialized{false};
    nrx::gfx::DxContext* dxContext{nullptr};
    std::filesystem::path activeModelPath;
    float activeScoreThreshold{0.45F};
    ImagePreprocessor preprocessor;
    OrtSessionManager sessionManager;
    Postprocessor postprocessor;
};

InferenceEngine::InferenceEngine() : impl(std::make_unique<Impl>()) {}

InferenceEngine::~InferenceEngine() = default;

auto InferenceEngine::init(nrx::gfx::DxContext* dxContext, const std::filesystem::path& modelPath)
    -> std::expected<void, InferenceError> {
    return impl->init(dxContext, modelPath);
}

auto InferenceEngine::execute(ID3D12Resource* inputTexture, D3D12_RESOURCE_STATES currentState)
    -> std::expected<DetectionResults, InferenceError> {
    return impl->execute(inputTexture, currentState);
}

auto InferenceEngine::update(const std::filesystem::path& modelPath, float confidenceThreshold)
    -> bool {
    return impl->update(modelPath, confidenceThreshold);
}

auto InferenceEngine::reinitialize() -> bool { return impl->reinitialize(); }

void InferenceEngine::setScoreThreshold(float value) { impl->setScoreThreshold(value); }

void InferenceEngine::reset() { impl->reset(); }

} // namespace nrx::inference
