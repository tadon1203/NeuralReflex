if(NOT DEFINED INPUT_FILE)
    message(FATAL_ERROR "INPUT_FILE is required")
endif()

if(NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "OUTPUT_FILE is required")
endif()

if(NOT DEFINED SYMBOL_BASENAME)
    set(SYMBOL_BASENAME "PreprocessCsHlsl")
endif()

file(READ "${INPUT_FILE}" shaderSource)

# Escape for C++ string literal.
string(REPLACE "\\" "\\\\" shaderSource "${shaderSource}")
string(REPLACE "\"" "\\\"" shaderSource "${shaderSource}")
string(REPLACE "\r\n" "\n" shaderSource "${shaderSource}")
string(REPLACE "\r" "\n" shaderSource "${shaderSource}")
string(REPLACE "\n" "\\n\"\n    \"" shaderSource "${shaderSource}")

get_filename_component(outputDirectory "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${outputDirectory}")

file(WRITE "${OUTPUT_FILE}" "#pragma once\n\n")
file(APPEND "${OUTPUT_FILE}" "#include <cstddef>\n\n")
file(APPEND "${OUTPUT_FILE}" "namespace nrx::inference::shaders {\n\n")
file(APPEND "${OUTPUT_FILE}" "inline constexpr char k${SYMBOL_BASENAME}Source[] =\n")
file(APPEND "${OUTPUT_FILE}" "    \"${shaderSource}\";\n\n")
file(APPEND "${OUTPUT_FILE}" "inline constexpr std::size_t k${SYMBOL_BASENAME}SourceLength =\n")
file(APPEND "${OUTPUT_FILE}" "    sizeof(k${SYMBOL_BASENAME}Source) - 1;\n\n")
file(APPEND "${OUTPUT_FILE}" "} // namespace nrx::inference::shaders\n")
