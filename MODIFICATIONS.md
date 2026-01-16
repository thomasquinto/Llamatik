# Llamatik Local Modifications

This document describes the local modifications made to the Llamatik library for use in the Edge AI App Factory project. These changes have not been upstreamed to the original Llamatik repository.

## Overview

The modifications fall into several categories:
1. **Build Configuration** - NDK version, CMake settings, Vulkan support
2. **ARM CPU Optimizations** - Better performance on modern Android devices
3. **GPU Acceleration** - Optional Vulkan support for Android
4. **Thread Safety** - Mutex and safe shutdown handling
5. **Prompt Formatting** - Model-agnostic prompt format
6. **Default Parameters** - Aligned with app defaults

---

## 1. Build Configuration (`library/build.gradle.kts`)

### NDK Version
```kotlin
ndkVersion = "26.1.10909125"
```
**Why:** Explicitly set NDK version to ensure consistent builds across development machines.

### CMake Arguments
```kotlin
externalNativeBuild {
    cmake {
        arguments += "-DCMAKE_BUILD_TYPE=Release"
        arguments += "-DANDROID_PLATFORM=android-29"
        arguments += "-DLLAMATIK_ENABLE_VULKAN=${if (enableVulkan) "ON" else "OFF"}"
    }
}
```
**Why:**
- `Release` build type ensures optimized native code even in debug APKs
- API 29 required for Vulkan 1.1 support (vkGetPhysicalDeviceFeatures2)
- Vulkan is opt-in via `-Pllamatik.vulkan=true` gradle property

### JNI Packaging
```kotlin
jniLibs {
    excludes += "**/libOpenCL.so"
}
```
**Why:** Exclude OpenCL stub library - it's only used for linking. At runtime, the app uses the system's `/vendor/lib64/libOpenCL.so`.

---

## 2. ARM CPU Optimizations (`library/src/commonMain/cpp/CMakeLists.txt`)

### Architecture Target
```cmake
if(ANDROID AND CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "ARM architecture" FORCE)
endif()
```
**Why:** Enables ARM dot product and FP16 instructions supported by ~95% of Android devices (2018+):
- Snapdragon 845+
- Exynos 9810+
- Google Tensor G1+
- Dimensity 800+

This provides significant performance improvements for quantized model inference.

### Removed Baseline Fallback
Removed the old `-march=armv8-a` fallback that was limiting performance on modern devices.

---

## 3. GPU Acceleration (`CMakeLists.txt` + `llama_jni.cpp`)

### Vulkan Option
```cmake
option(LLAMATIK_ENABLE_VULKAN "Enable Vulkan GPU acceleration for Android" OFF)
```
**Why:** Vulkan GPU acceleration is available but disabled by default because:
- CPU with ARM optimizations is often faster than Vulkan on many devices
- Vulkan adds complexity and potential compatibility issues
- Can be enabled with `-Pllamatik.vulkan=true` for testing

### GPU Layer Offloading
```cpp
mparams.n_gpu_layers = 99;
```
**Why:** When Vulkan is enabled, offload all model layers to GPU for maximum acceleration.

---

## 4. Thread Safety (`llama_jni.cpp` + iOS `llama_embed.cpp`)

### Generation Mutex
```cpp
static std::mutex g_generation_mutex;
// ...
std::lock_guard<std::mutex> lock(g_generation_mutex);
```
**Why:** Serialize generation calls to prevent KV cache corruption when multiple requests arrive simultaneously.

### Generation Progress Flag
```cpp
static std::atomic<bool> g_generation_in_progress{false};
```
**Why:** Track whether generation is active for safe shutdown handling.

### Safe Shutdown
```cpp
void shutdown() {
    g_cancel_requested.store(true);
    // Wait for generation to complete (with 5s timeout)
    while (g_generation_in_progress.load() && wait_count < max_wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // Then free resources...
}
```
**Why:** Prevents use-after-free crashes when the app is closed while generation is in progress.

---

## 5. Prompt Formatting

### Model-Agnostic Format
**Before (Gemma-specific):**
```cpp
oss << "<start_of_turn>system\n" << sys << "\n<end_of_turn>\n"
    << "<start_of_turn>user\n" << user_msg << "\n<end_of_turn>\n"
    << "<start_of_turn>model\n";
```

**After (Plain format):**
```cpp
if (!trim(system_msg).empty()) {
    oss << "Instructions: " << system_msg << "\n\n";
}
if (!trim(context_block).empty()) {
    oss << "Context:\n" << context_block << "\n\n";
}
oss << "Question:\n" << user_msg << "\n\nAnswer:\n";
```

**Why:** The Gemma-specific chat template (`<start_of_turn>`, etc.) caused issues with other models like Llama. The plain format works with any model and lets the Kotlin layer handle model-specific formatting if needed.

### Kotlin Bridge Simplification
Removed `buildChatPrompt()` from `LlamaBridge.android.kt` and `LlamaBridge.jvm.kt` - now delegates directly to native code.

---

## 6. Default Parameters

### Aligned with App Defaults
```cpp
// Before
static std::atomic<float> g_temperature = 0.55f;
static std::atomic<float> g_top_p = 0.80f;
static std::atomic<int>   g_top_k = 20;
static std::atomic<int>   g_max_new_tokens = 640;

// After
static std::atomic<float> g_temperature = 0.7f;
static std::atomic<float> g_top_p = 0.95f;
static std::atomic<int>   g_top_k = 40;
static std::atomic<int>   g_max_new_tokens = 512;
```
**Why:** Align native defaults with the app's UI defaults for consistent behavior.

---

## 7. Logging Improvements (`llama_jni.cpp`)

### Log Callback
```cpp
static void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    // Route llama.cpp logs to Android logcat
}
```
**Why:** Makes llama.cpp internal logs visible in Android logcat for debugging.

### Backend Discovery Logging
```cpp
size_t num_devices = ggml_backend_dev_count();
for (size_t i = 0; i < num_devices; i++) {
    // Log device name, description, type (CPU/GPU/etc.)
}
```
**Why:** Helps diagnose which compute backends are available on the device.

---

## Files Modified

| File | Changes |
|------|---------|
| `library/build.gradle.kts` | NDK version, CMake args, JNI excludes |
| `library/src/commonMain/cpp/CMakeLists.txt` | ARM optimizations, Vulkan option |
| `library/src/commonMain/cpp/llama_embed.cpp` | GPU layers, logging |
| `library/src/commonMain/cpp/llama_jni.cpp` | Thread safety, prompt format, logging, defaults |
| `library/src/iosMain/cpp/llama_embed.cpp` | Thread safety, prompt format, defaults |
| `library/src/androidMain/kotlin/.../LlamaBridge.android.kt` | Removed Gemma template |
| `library/src/jvmMain/kotlin/.../LlamaBridge.jvm.kt` | Removed Gemma template |

---

## Future Considerations

1. **Upstream these changes** - Consider contributing thread safety and model-agnostic prompt changes back to Llamatik
2. **Vulkan testing** - More testing needed to determine when Vulkan is beneficial vs CPU
3. **ARM i8mm support** - Could add `+i8mm` for even newer devices (2020+) with runtime detection

