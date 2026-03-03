#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace nrx::inference {

struct DetectionResult {
    float x;
    float y;
    float w;
    float h;
    float score;
    std::uint32_t classId;
};

struct Resolution {
    std::uint32_t width;
    std::uint32_t height;
};

using DetectionResults = std::vector<DetectionResult>;

enum class InferenceError : std::uint8_t {
    InvalidArguments,
    NotInitialized,
    PreprocessFailed,
    SessionInitFailed,
    IoBindingFailed,
    RunFailed,
    PostprocessFailed,
    DeviceLost,
};

[[nodiscard]] inline auto inferenceErrorToString(InferenceError error) -> std::string_view {
    switch (error) {
    case InferenceError::InvalidArguments:
        return "InvalidArguments";
    case InferenceError::NotInitialized:
        return "NotInitialized";
    case InferenceError::PreprocessFailed:
        return "PreprocessFailed";
    case InferenceError::SessionInitFailed:
        return "SessionInitFailed";
    case InferenceError::IoBindingFailed:
        return "IoBindingFailed";
    case InferenceError::RunFailed:
        return "RunFailed";
    case InferenceError::PostprocessFailed:
        return "PostprocessFailed";
    case InferenceError::DeviceLost:
        return "DeviceLost";
    }

    return "UnknownInferenceError";
}

} // namespace nrx::inference
