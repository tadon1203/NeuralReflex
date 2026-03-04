#include "nrx/inference/ort_session_manager.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime/core/providers/dml/dml_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <DirectML.h>
#include <Windows.h>
#include <d3d12.h>
#include <winrt/base.h>

#include "nrx/gfx/dx_context.hpp"
#include "nrx/inference/types.hpp"
#include "nrx/utils/dx_helper.hpp"
#include "nrx/utils/logger.hpp"

namespace nrx::inference {

namespace {

constexpr Resolution kInputResolution{
    .width = 640,
    .height = 640,
};
constexpr std::uint32_t kInputChannels = 3;

} // namespace

class OrtSessionManager::Impl {
  public:
    auto init(nrx::gfx::DxContext* context, const std::filesystem::path& modelPath)
        -> std::expected<void, InferenceError> {
        reset();

        if (context == nullptr || modelPath.empty()) {
            return std::unexpected(InferenceError::InvalidArguments);
        }
        if (context->checkDeviceLost()) {
            return std::unexpected(InferenceError::DeviceLost);
        }

        dxContext = context;

        try {
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NeuralReflex");
            sessionOptions = std::make_unique<Ort::SessionOptions>();
            sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            const OrtApi& api = Ort::GetApi();
            const void* providerApi = nullptr;
            OrtStatus* providerStatus =
                api.GetExecutionProviderApi("DML", ORT_API_VERSION, &providerApi);
            if (providerStatus != nullptr || providerApi == nullptr) {
                const char* message = providerStatus != nullptr
                                          ? api.GetErrorMessage(providerStatus)
                                          : "provider API pointer is null";
                NRX_ERROR("OrtSessionManager failed to query DML provider API: {}", message);
                if (providerStatus != nullptr) {
                    api.ReleaseStatus(providerStatus);
                }
                reset();
                return std::unexpected(InferenceError::SessionInitFailed);
            }
            dmlApi = static_cast<const OrtDmlApi*>(providerApi);

            auto* d12Device = context->getD12Device();
            auto* d12Queue = context->getD12Queue();
            if (d12Device == nullptr || d12Queue == nullptr) {
                reset();
                return std::unexpected(InferenceError::SessionInitFailed);
            }

            const HRESULT createDmlDeviceHr = DMLCreateDevice(
                d12Device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(dmlDevice.put()));
            if (FAILED(createDmlDeviceHr)) {
                const auto removedReason = d12Device->GetDeviceRemovedReason();
                NRX_ERROR("OrtSessionManager failed to create IDMLDevice: {}",
                          nrx::utils::DxHelper::getErrorString(
                              static_cast<std::int32_t>(createDmlDeviceHr)));
                NRX_ERROR(
                    "OrtSessionManager D3D12 device removed reason: {}",
                    nrx::utils::DxHelper::getErrorString(static_cast<std::int32_t>(removedReason)));
                reset();
                return std::unexpected(InferenceError::SessionInitFailed);
            }

            OrtStatus* dmlStatus = dmlApi->SessionOptionsAppendExecutionProvider_DML1(
                *sessionOptions, dmlDevice.get(), d12Queue);
            if (dmlStatus != nullptr) {
                const char* message = Ort::GetApi().GetErrorMessage(dmlStatus);
                NRX_ERROR("OrtSessionManager DML EP setup failed: {}", message);
                Ort::GetApi().ReleaseStatus(dmlStatus);
                reset();
                return std::unexpected(InferenceError::SessionInitFailed);
            }

            const std::wstring modelPathWide = modelPath.wstring();
            session = std::make_unique<Ort::Session>(*env, modelPathWide.c_str(), *sessionOptions);
            ioBinding = std::make_unique<Ort::IoBinding>(*session);

            Ort::AllocatorWithDefaultOptions allocator;
            inputName = session->GetInputNameAllocated(0, allocator).get();
            outputName = session->GetOutputNameAllocated(0, allocator).get();

            const auto inputType = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            inputShape = inputType.GetShape();
            if (inputShape.size() >= 4) {
                inputShape[inputShape.size() - 1] = static_cast<int64_t>(kInputResolution.width);
                inputShape[inputShape.size() - 2] = static_cast<int64_t>(kInputResolution.height);
            } else {
                inputShape = {1, static_cast<int64_t>(kInputChannels),
                              static_cast<int64_t>(kInputResolution.height),
                              static_cast<int64_t>(kInputResolution.width)};
            }

            const auto outputType = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
            outputShapeData = outputType.GetShape();
            const auto outputElementCount = outputType.GetElementCount();
            outputBytes = outputElementCount * sizeof(float);
            if (outputBytes == 0) {
                reset();
                return std::unexpected(InferenceError::SessionInitFailed);
            }

            const auto outputBufferResult = nrx::utils::DxHelper::createBuffer(
                d12Device, outputBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_HEAP_TYPE_DEFAULT);
            if (!outputBufferResult) {
                NRX_ERROR("OrtSessionManager output resource creation failed: {}",
                          nrx::utils::DxHelper::getErrorString(outputBufferResult.error()));
                reset();
                return std::unexpected(InferenceError::SessionInitFailed);
            }
            outputResource = outputBufferResult.value();

            OrtStatus* createOutputAllocationStatus = dmlApi->CreateGPUAllocationFromD3DResource(
                outputResource.get(), &outputDmlAllocation);
            if (createOutputAllocationStatus != nullptr || outputDmlAllocation == nullptr) {
                const char* message =
                    createOutputAllocationStatus != nullptr
                        ? Ort::GetApi().GetErrorMessage(createOutputAllocationStatus)
                        : "CreateGPUAllocationFromD3DResource returned null output allocation";
                NRX_ERROR("OrtSessionManager output allocation failed: {}", message);
                if (createOutputAllocationStatus != nullptr) {
                    Ort::GetApi().ReleaseStatus(createOutputAllocationStatus);
                }
                reset();
                return std::unexpected(InferenceError::SessionInitFailed);
            }

            dmlMemoryInfo = std::make_unique<Ort::MemoryInfo>("DML", OrtMemoryInfoDeviceType_GPU, 0,
                                                              0, OrtDeviceMemoryType_DEFAULT, 0,
                                                              OrtDeviceAllocator);

            initialized = true;
            return {};
        } catch (const Ort::Exception& e) {
            NRX_ERROR("OrtSessionManager init failed: {}", e.what());
            reset();
            return std::unexpected(InferenceError::SessionInitFailed);
        }
    }

    auto run(ID3D12Resource* inputTensorResource)
        -> std::expected<OrtSessionOutput, InferenceError> {
        if (!initialized || dxContext == nullptr || session == nullptr || ioBinding == nullptr ||
            dmlMemoryInfo == nullptr || dmlApi == nullptr || outputResource == nullptr ||
            outputDmlAllocation == nullptr) {
            return std::unexpected(InferenceError::NotInitialized);
        }
        if (dxContext->checkDeviceLost()) {
            return std::unexpected(InferenceError::DeviceLost);
        }
        if (inputTensorResource == nullptr) {
            return std::unexpected(InferenceError::InvalidArguments);
        }

        void* dmlAllocation = nullptr;
        OrtStatus* createAllocationStatus =
            dmlApi->CreateGPUAllocationFromD3DResource(inputTensorResource, &dmlAllocation);
        if (createAllocationStatus != nullptr || dmlAllocation == nullptr) {
            const char* message =
                createAllocationStatus != nullptr
                    ? Ort::GetApi().GetErrorMessage(createAllocationStatus)
                    : "CreateGPUAllocationFromD3DResource returned null allocation";
            NRX_ERROR("OrtSessionManager input allocation failed: {}", message);
            if (createAllocationStatus != nullptr) {
                Ort::GetApi().ReleaseStatus(createAllocationStatus);
            }
            return std::unexpected(InferenceError::IoBindingFailed);
        }

        const auto inputBytes = static_cast<std::uint64_t>(kInputResolution.width) *
                                kInputResolution.height * kInputChannels * sizeof(float);

        try {
            ioBinding->ClearBoundInputs();
            ioBinding->ClearBoundOutputs();

            Ort::Value inputValue = Ort::Value::CreateTensor(
                *dmlMemoryInfo, dmlAllocation, static_cast<size_t>(inputBytes), inputShape.data(),
                inputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            Ort::Value outputValue = Ort::Value::CreateTensor(
                *dmlMemoryInfo, outputDmlAllocation, static_cast<size_t>(outputBytes),
                outputShapeData.data(), outputShapeData.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

            ioBinding->BindInput(inputName.c_str(), inputValue);
            ioBinding->BindOutput(outputName.c_str(), outputValue);

            Ort::RunOptions runOptions;
            session->Run(runOptions, *ioBinding);
            ioBinding->SynchronizeOutputs();
            dmlApi->FreeGPUAllocation(dmlAllocation);
            outputResourceState = D3D12_RESOURCE_STATE_COMMON;
            return OrtSessionOutput{
                .resource = outputResource.get(),
                .currentState = outputResourceState,
            };
        } catch (const Ort::Exception& e) {
            NRX_ERROR("OrtSessionManager run failed: {}", e.what());
            dmlApi->FreeGPUAllocation(dmlAllocation);
            return std::unexpected(InferenceError::RunFailed);
        }
    }

    [[nodiscard]] auto getInputResolution() const -> Resolution { return kInputResolution; }

    [[nodiscard]] auto outputShape() const -> std::span<const int64_t> {
        return {outputShapeData.data(), outputShapeData.size()};
    }

    void reset() {
        initialized = false;
        inputShape.clear();
        outputShapeData.clear();
        outputBytes = 0;
        inputName.clear();
        outputName.clear();
        if (dmlApi != nullptr && outputDmlAllocation != nullptr) {
            OrtStatus* freeOutputStatus = dmlApi->FreeGPUAllocation(outputDmlAllocation);
            if (freeOutputStatus != nullptr) {
                const char* message = Ort::GetApi().GetErrorMessage(freeOutputStatus);
                NRX_WARN("OrtSessionManager output allocation release failed: {}", message);
                Ort::GetApi().ReleaseStatus(freeOutputStatus);
            }
        }
        outputDmlAllocation = nullptr;
        outputResource = nullptr;
        outputResourceState = D3D12_RESOURCE_STATE_COMMON;
        dmlApi = nullptr;
        dmlDevice = nullptr;
        ioBinding.reset();
        session.reset();
        sessionOptions.reset();
        env.reset();
        dmlMemoryInfo.reset();
        dxContext = nullptr;
    }

  private:
    bool initialized{false};
    nrx::gfx::DxContext* dxContext{nullptr};

    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::IoBinding> ioBinding;

    std::unique_ptr<Ort::MemoryInfo> dmlMemoryInfo;

    const OrtDmlApi* dmlApi{nullptr};
    winrt::com_ptr<IDMLDevice> dmlDevice;
    winrt::com_ptr<ID3D12Resource> outputResource;
    D3D12_RESOURCE_STATES outputResourceState{D3D12_RESOURCE_STATE_COMMON};
    void* outputDmlAllocation{nullptr};

    std::string inputName;
    std::string outputName;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShapeData;
    std::uint64_t outputBytes{0};
};

OrtSessionManager::OrtSessionManager() : impl(std::make_unique<Impl>()) {}

OrtSessionManager::~OrtSessionManager() = default;

auto OrtSessionManager::init(nrx::gfx::DxContext* dxContext, const std::filesystem::path& modelPath)
    -> std::expected<void, InferenceError> {
    return impl->init(dxContext, modelPath);
}

auto OrtSessionManager::run(ID3D12Resource* inputTensorResource)
    -> std::expected<OrtSessionOutput, InferenceError> {
    return impl->run(inputTensorResource);
}

auto OrtSessionManager::getInputResolution() const -> Resolution {
    return impl->getInputResolution();
}

auto OrtSessionManager::outputShape() const -> std::span<const int64_t> {
    return impl->outputShape();
}

void OrtSessionManager::reset() { impl->reset(); }

} // namespace nrx::inference
