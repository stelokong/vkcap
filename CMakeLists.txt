cmake_minimum_required(VERSION 3.24)
project(vkcap VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTING "Build unit and GPU integration tests" ON)

find_package(PkgConfig REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

pkg_check_modules(VULKAN REQUIRED IMPORTED_TARGET vulkan)
pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)
pkg_check_modules(PIPEWIRE REQUIRED IMPORTED_TARGET libpipewire-0.3)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec>=61.0
    libavformat>=61.0
    libavutil>=59.0
)

set(GENERATED_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
file(MAKE_DIRECTORY "${GENERATED_SHADER_DIR}")

function(add_embedded_shader SHADER_SRC VARIANT_NAME DEFINE_FLAG SYMBOL_NAME OUT_HEADER)
    set(SPV_FILE "${GENERATED_SHADER_DIR}/${VARIANT_NAME}.spv")
    set(HDR_FILE "${GENERATED_SHADER_DIR}/${VARIANT_NAME}_spv.hpp")

    add_custom_command(
        OUTPUT "${SPV_FILE}"
        COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -O "${DEFINE_FLAG}" "${SHADER_SRC}" -o "${SPV_FILE}"
        DEPENDS "${SHADER_SRC}"
        COMMENT "Compiling shader ${VARIANT_NAME}.spv (${DEFINE_FLAG})"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${HDR_FILE}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.py"
                "${SPV_FILE}" "${HDR_FILE}" "${SYMBOL_NAME}"
        DEPENDS "${SPV_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spv.py"
        COMMENT "Embedding SPIR-V ${VARIANT_NAME}.spv -> ${VARIANT_NAME}_spv.hpp"
        VERBATIM
    )

    set(${OUT_HEADER} "${HDR_FILE}" PARENT_SCOPE)
endfunction()

add_embedded_shader(
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rgb_to_yuv.comp"
    "rgb_to_yuv_nv12"
    "-DP010=0"
    "kRgbToYuvNv12Spv"
    RGB_TO_YUV_NV12_HEADER
)

add_embedded_shader(
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rgb_to_yuv.comp"
    "rgb_to_yuv_p010"
    "-DP010=1"
    "kRgbToYuvP010Spv"
    RGB_TO_YUV_P010_HEADER
)

add_library(vkcap_core STATIC
    src/audio_capture.cpp
    src/cli.cpp
    src/color_math.cpp
    src/interactive.cpp
    src/user_config.cpp
    src/privileges.cpp
    src/kms_capture.cpp
    src/vulkan_compute.cpp
    src/video_encoder.cpp
    src/pipeline.cpp
    "${RGB_TO_YUV_NV12_HEADER}"
    "${RGB_TO_YUV_P010_HEADER}"
)

target_include_directories(vkcap_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${GENERATED_SHADER_DIR}"
)

target_link_libraries(vkcap_core
    PUBLIC
        PkgConfig::VULKAN
        PkgConfig::LIBDRM
        PkgConfig::PIPEWIRE
        PkgConfig::FFMPEG
)

target_compile_options(vkcap_core PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-missing-field-initializers
    $<$<CONFIG:Release>:-fno-math-errno -fno-trapping-math>
)

add_executable(vkcap src/main.cpp)
target_link_libraries(vkcap PRIVATE vkcap_core)
target_compile_options(vkcap PRIVATE -Wall -Wextra -Wpedantic)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET vkcap_core vkcap PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(TARGETS vkcap RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    enable_testing()
    add_executable(vkcap_tests tests/test_main.cpp)
    target_link_libraries(vkcap_tests PRIVATE vkcap_core)
    target_compile_options(vkcap_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME vkcap_tests COMMAND vkcap_tests)
endif()
