#include "nrx/inference/inference_engine.hpp"

#include <expected>
#include <filesystem>
#include <memory>

#include "nrx/gfx/dx_context.hpp"
#include "nrx/inference/image_preprocessor.hpp"
#include "nrx/inference/ort_session_manager.hpp"
#include "nrx/inference/postprocessor.hpp"
#include "nrx/inference/types.hpp"

namespace nrx::inference {

class InferenceEngine::Impl {
  public:
    auto init(nrx::gfx::DxContext* dxContext, const std::filesystem::path& modelPath)
        -> std::expected<void, InferenceError> {
        reset();

        if (dxContext == nullptr || modelPath.empty()) {
            return std::unexpected(InferenceError::InvalidArguments);
        }
        if (dxContext->checkDeviceLost()) {
            return std::unexpected(InferenceError::DeviceLost);
        }

        if (const auto preprocessResult = preprocessor.init(dxContext); !preprocessResult) {
            reset();
            return std::unexpected(preprocessResult.error());
        }

        if (const auto sessionResult = sessionManager.init(dxContext, modelPath); !sessionResult) {
            reset();
            return std::unexpected(sessionResult.error());
        }

        if (const auto postprocessorResult = postprocessor.init(
                dxContext, sessionManager.outputShape(), sessionManager.getInputResolution());
            !postprocessorResult) {
            reset();
            return std::unexpected(postprocessorResult.error());
        }

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

        const auto preprocessed = preprocessor.preprocess(inputTexture, currentState);
        if (!preprocessed) {
            return std::unexpected(preprocessed.error());
        }

        const auto rawOutputResource = sessionManager.run(preprocessed.value());
        if (!rawOutputResource) {
            return std::unexpected(rawOutputResource.error());
        }

        const auto postDispatchResult = postprocessor.dispatch(rawOutputResource.value());
        if (!postDispatchResult) {
            return std::unexpected(postDispatchResult.error());
        }

        return postprocessor.readbackFinalResults();
    }

    void reset() {
        initialized = false;
        postprocessor.reset();
        sessionManager.reset();
        preprocessor.reset();
    }

  private:
    bool initialized{false};
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

void InferenceEngine::reset() { impl->reset(); }

} // namespace nrx::inference
