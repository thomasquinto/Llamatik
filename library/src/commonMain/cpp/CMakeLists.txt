cmake_minimum_required(VERSION 3.22.1)
project(llama_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# REMOVE these (they enabled advanced ARM features and caused SIGILL)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.7-a")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.7-a")

if(ANDROID)
    add_link_options("-Wl,-z,max-page-size=16384")
endif()

set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--section-start=.note.gnu.build-id=0x4000 -Wl,--no-rosegment"
)

# ARM CPU optimization for Android
# armv8.2-a+dotprod+fp16 is supported by ~95% of Android devices (2018+)
# This includes: Snapdragon 845+, Exynos 9810+, Tensor G1+, Dimensity 800+
if(ANDROID AND CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "ARM architecture" FORCE)
endif()

# GPU acceleration option for Android
# Default: OFF (CPU with ARM optimizations is often faster than Vulkan on many devices)
# To enable Vulkan: pass -DLLAMATIK_ENABLE_VULKAN=ON to CMake
option(LLAMATIK_ENABLE_VULKAN "Enable Vulkan GPU acceleration for Android" OFF)

if(ANDROID)
    if(LLAMATIK_ENABLE_VULKAN)
        set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan GPU acceleration" FORCE)

        # Use system Vulkan headers (from Vulkan SDK) if available
        if(EXISTS "/usr/local/include/vulkan/vulkan.hpp")
            include_directories(SYSTEM "/usr/local/include")
            message(STATUS "Using system Vulkan C++ headers from /usr/local/include")
        endif()

        message(STATUS "Vulkan GPU acceleration ENABLED")
    else()
        set(GGML_VULKAN OFF CACHE BOOL "Disable Vulkan GPU acceleration" FORCE)
        message(STATUS "Using CPU-only with ARM optimizations (dotprod+fp16)")
    endif()
endif()

add_compile_definitions(GGML_MAX_NAME=128)
add_subdirectory("${CMAKE_SOURCE_DIR}/../../../../llama.cpp/" "${CMAKE_BINARY_DIR}/llama")

add_library(llama_jni SHARED
        llama_jni.cpp
        llama_embed.cpp
        ${LLAMA_SOURCES}
)

find_library(log-lib log)

target_include_directories(llama_jni PRIVATE
        ${CMAKE_SOURCE_DIR}/../c_interop/include
)

target_link_libraries(llama_jni llama ${log-lib})