cmake_minimum_required(VERSION 4.0.0)
project(NeuralReflex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(spdlog)

set(DIRECTX_HEADERS_BUILD_TESTS OFF CACHE BOOL "Disable DirectX-Headers tests" FORCE)
FetchContent_Declare(
    directx_headers
    GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers.git
    GIT_TAG        v1.619.0
)
FetchContent_MakeAvailable(directx_headers)

add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)

set(PREPROCESS_SHADER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/preprocess_cs.hlsl"
)
set(PREPROCESS_SHADER_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/inference/preprocess_cs_embedded.hpp"
)
set(POST_DECODE_SHADER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/post_decode_filter_cs.hlsl"
)
set(POST_DECODE_SHADER_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/inference/post_decode_filter_cs_embedded.hpp"
)
set(POST_NMS_SHADER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/post_nms_cs.hlsl"
)
set(POST_NMS_SHADER_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/inference/post_nms_cs_embedded.hpp"
)
set(POST_COMPACT_SHADER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/post_compact_cs.hlsl"
)
set(POST_COMPACT_SHADER_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/inference/post_compact_cs_embedded.hpp"
)
set(SHADER_EMBED_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_shader_header.cmake"
)

add_custom_command(
    OUTPUT "${PREPROCESS_SHADER_HEADER}"
    COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${PREPROCESS_SHADER_SOURCE}
            -DOUTPUT_FILE=${PREPROCESS_SHADER_HEADER}
            -DSYMBOL_BASENAME=PreprocessCsHlsl
            -P "${SHADER_EMBED_SCRIPT}"
    DEPENDS "${PREPROCESS_SHADER_SOURCE}" "${SHADER_EMBED_SCRIPT}"
    VERBATIM
)

add_custom_command(
    OUTPUT "${POST_DECODE_SHADER_HEADER}"
    COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${POST_DECODE_SHADER_SOURCE}
            -DOUTPUT_FILE=${POST_DECODE_SHADER_HEADER}
            -DSYMBOL_BASENAME=PostDecodeFilterCsHlsl
            -P "${SHADER_EMBED_SCRIPT}"
    DEPENDS "${POST_DECODE_SHADER_SOURCE}" "${SHADER_EMBED_SCRIPT}"
    VERBATIM
)

add_custom_command(
    OUTPUT "${POST_NMS_SHADER_HEADER}"
    COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${POST_NMS_SHADER_SOURCE}
            -DOUTPUT_FILE=${POST_NMS_SHADER_HEADER}
            -DSYMBOL_BASENAME=PostNmsCsHlsl
            -P "${SHADER_EMBED_SCRIPT}"
    DEPENDS "${POST_NMS_SHADER_SOURCE}" "${SHADER_EMBED_SCRIPT}"
    VERBATIM
)

add_custom_command(
    OUTPUT "${POST_COMPACT_SHADER_HEADER}"
    COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${POST_COMPACT_SHADER_SOURCE}
            -DOUTPUT_FILE=${POST_COMPACT_SHADER_HEADER}
            -DSYMBOL_BASENAME=PostCompactCsHlsl
            -P "${SHADER_EMBED_SCRIPT}"
    DEPENDS "${POST_COMPACT_SHADER_SOURCE}" "${SHADER_EMBED_SCRIPT}"
    VERBATIM
)

add_custom_target(generate_embedded_shaders
    DEPENDS
    "${PREPROCESS_SHADER_HEADER}"
    "${POST_DECODE_SHADER_HEADER}"
    "${POST_NMS_SHADER_HEADER}"
    "${POST_COMPACT_SHADER_HEADER}"
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/core/application.cpp
    src/core/config_manager.cpp
    src/core/platform_win.cpp
    src/utils/logger.cpp
    src/utils/dx_helper.cpp
    src/gfx/dx_context.cpp
    src/gfx/screen_capturer.cpp
    src/gfx/gfx_bridge.cpp
    src/inference/inference_engine.cpp
    src/inference/image_preprocessor.cpp
    src/inference/ort_session_manager.cpp
    src/inference/postprocessor.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/1.25.0/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/1.25.0/include/onnxruntime/core/session
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    NRX_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_dependencies(${PROJECT_NAME} generate_embedded_shaders)
set_source_files_properties(
    src/inference/image_preprocessor.cpp
    PROPERTIES OBJECT_DEPENDS "${PREPROCESS_SHADER_HEADER}"
)
set_source_files_properties(
    src/inference/postprocessor.cpp
    PROPERTIES OBJECT_DEPENDS
    "${POST_DECODE_SHADER_HEADER};${POST_NMS_SHADER_HEADER};${POST_COMPACT_SHADER_HEADER}"
)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/1.25.0")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib/win-x64/RelWithDebInfo")

target_link_libraries(${PROJECT_NAME} PRIVATE
    spdlog::spdlog
    Microsoft::DirectX-Headers
    d3d11
    d3d12
    d3dcompiler
    dxgi
    WindowsApp
    "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib"
    "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_dml.lib"
    "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_shared.lib"
)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/DirectML.dll"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMENT "Copying ONNX Runtime and DirectML runtime DLLs"
    VERBATIM
)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(COMPILE_COMMANDS_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
    set(COMPILE_COMMANDS_DEST   "${CMAKE_SOURCE_DIR}/build/compile_commands.json")

    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/build")

    # Best-effort copy during configure/generate if source already exists.
    if(EXISTS "${COMPILE_COMMANDS_SOURCE}")
        file(COPY_FILE "${COMPILE_COMMANDS_SOURCE}" "${COMPILE_COMMANDS_DEST}" ONLY_IF_DIFFERENT)
    endif()

    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${COMPILE_COMMANDS_SOURCE}"
                "${COMPILE_COMMANDS_DEST}"
        COMMENT "Copying compile_commands.json to build/"
        VERBATIM
    )
endif()
