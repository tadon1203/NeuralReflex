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

add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/core/application.cpp
    src/utils/logger.cpp
    src/utils/dx_helper.cpp
    src/gfx/dx_context.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    spdlog::spdlog
    d3d11
    d3d12
    dxgi
    WindowsApp
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
