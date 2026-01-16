#include <jni.h>
#include "llama.h"
#include "llama_jni.h"
#include "ggml-backend.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <cstring>   // strlen, memcpy
#include <cctype>    // tolower, isalpha, isdigit
#include <cstdlib>   // malloc, free
#include <string_view>
#include <vector>
#include <atomic>
#include <cstdio>
#include <thread>
#include <chrono>
#include <mutex>

// ===================================================================================
//                              PLATFORM LOGGING
// ===================================================================================
//
// Android uses logcat; Desktop uses stderr.
// This file is shared for Android + Desktop builds.

#if defined(__ANDROID__)
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "LlamaBridge", __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "LlamaBridge", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  "LlamaBridge", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "LlamaBridge", __VA_ARGS__)
#else
static void log_stderr(const char* level, const char* fmt, ...) {
        std::fprintf(stderr, "[LlamaBridge][%s] ", level);
        va_list args;
        va_start(args, fmt);
        std::vfprintf(stderr, fmt, args);
        va_end(args);
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
    #define LOGI(...) log_stderr("I", __VA_ARGS__)
    #define LOGD(...) log_stderr("D", __VA_ARGS__)
    #define LOGW(...) log_stderr("W", __VA_ARGS__)
    #define LOGE(...) log_stderr("E", __VA_ARGS__)
#endif

// ===================================================================================
//                              GLOBAL STATE (this TU)
// ===================================================================================

// Embeddings
static struct llama_model *emb_model = nullptr;
static struct llama_context *emb_ctx = nullptr;
static int emb_dim = 0;

// Text generation
static struct llama_model *gen_model = nullptr;
static struct llama_context *gen_ctx = nullptr;

// Backend lifetime
static bool g_backend_inited = false;

// Llama log callback to redirect logs to Android logcat
static void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (text == nullptr) return;
    // Remove trailing newline if present
    std::string msg(text);
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
    }
    if (msg.empty()) return;

    switch (level) {
        case GGML_LOG_LEVEL_ERROR:
            LOGE("llama: %s", msg.c_str());
            break;
        case GGML_LOG_LEVEL_WARN:
            LOGW("llama: %s", msg.c_str());
            break;
        case GGML_LOG_LEVEL_INFO:
            LOGI("llama: %s", msg.c_str());
            break;
        default:
            LOGD("llama: %s", msg.c_str());
            break;
    }
}

// Streaming cancel flag (for generateStream)
static std::atomic<bool> g_cancel_requested{false};

// Flag to track if generation is in progress (for safe shutdown)
static std::atomic<bool> g_generation_in_progress{false};

// Mutex to serialize generation calls (prevents KV cache corruption)
static std::mutex g_generation_mutex;

static std::atomic<float> g_temperature     = 0.7f;   // align with app default
static std::atomic<float> g_top_p          = 0.95f;  // align with app default
static std::atomic<int>   g_top_k          = 40;     // align with app default
static std::atomic<float> g_repeat_penalty = 1.10f;
static std::atomic<int>   g_max_new_tokens = 512;    // align with app default

// ===================================================================================
//                              SMALL HELPERS
// ===================================================================================

static inline std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static int tokenize_with_retry(const llama_vocab *vocab,
        const char *text,
        std::vector<llama_token> &tokens,
        bool add_bos,
        bool parse_special) {
    if (!text) return 0;
    const int text_len = (int) std::strlen(text);

    int n = llama_tokenize(vocab, text, text_len,
            tokens.data(),
            (int) tokens.size(),
            add_bos, parse_special);
    if (n < 0) {
        const int need = -n;
        if (need > 0) {
            tokens.resize(need);
            n = llama_tokenize(vocab, text, text_len,
                    tokens.data(),
                    (int) tokens.size(),
                    add_bos, parse_special);
        }
    }
    return n;
}

static void truncate_to_ctx(std::vector<llama_token> &tokens, int n_ctx, int reserve_tail) {
    if ((int) tokens.size() <= n_ctx - reserve_tail) return;
    const int keep = n_ctx - reserve_tail;
    std::vector<llama_token> out;
    out.reserve(keep);
    out.insert(out.end(), tokens.end() - keep, tokens.end());
    tokens.swap(out);
}

// ---------- Sanitizer (strong, used by non-streaming only) ----------
static void drop_lines_with_prefix(std::string &s, const char *prefix_lc) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0, line_start = 0;
    while (i <= s.size()) {
        if (i == s.size() || s[i] == '\n') {
            std::string_view line(s.data() + line_start, i - line_start);
            std::string line_lc = to_lower(std::string(line));
            if (!(line_lc.rfind(prefix_lc, 0) == 0)) {
                out.append(s.data() + line_start, i - line_start);
                if (i != s.size()) out.push_back('\n');
            }
            line_start = i + 1;
        }
        ++i;
    }
    s.swap(out);
}

// returns cleaned answer; fallback if too short or no alpha
static std::string sanitize_generation(std::string s) {
    if (s.empty()) return s;

    for (const char *stop: {"<end_of_turn>", "<|eot_id|>", "</s>"}) {
        size_t p = s.find(stop);
        if (p != std::string::npos) { s = s.substr(0, p); }
    }
    drop_lines_with_prefix(s, "<start_of_turn>");
    drop_lines_with_prefix(s, "<|start_header_id|>");
    drop_lines_with_prefix(s, "<|end_header_id|>");

    {
        std::string sl = to_lower(s);
        size_t qpos = sl.find("question:");
        if (qpos != std::string::npos) s = s.substr(0, qpos);
    }
    {
        std::string sl = to_lower(s);
        size_t cpos = sl.find("context:");
        if (cpos != std::string::npos) s = s.substr(0, cpos);
    }

    auto slice_after_tag = [&](const char *tag) -> bool {
        std::string low = to_lower(s);
        std::string t = to_lower(std::string(tag));
        size_t p = low.find(t);
        if (p != std::string::npos) {
            s = s.substr(p + std::strlen(tag));
            s = trim(s);
            return true;
        }
        return false;
    };
    (void) (slice_after_tag("ANSWER:") || slice_after_tag("FINAL_ANSWER:"));

    s = trim(s);

    auto strip_leading_noise = [](std::string &t) {
        auto ltrim_str = [&](const char *prefix) -> bool {
            size_t n = std::strlen(prefix);
            if (t.size() >= n && std::memcmp(t.data(), prefix, n) == 0) {
                t.erase(0, n);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                return true;
            }
            return false;
        };

        bool changed = true;
        while (changed) {
            changed = false;
            changed |= ltrim_str("• ");
            changed |= ltrim_str("- ");
            changed |= ltrim_str("* ");
            changed |= ltrim_str("> ");
            changed |= ltrim_str(u8"—");
            changed |= ltrim_str(u8"–");

            if (!t.empty() && (t[0] == ':' || t[0] == '-')) {
                t.erase(0, 1);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                changed = true;
            }

            if (t.size() >= 2 && std::isdigit(static_cast<unsigned char>(t[0])) &&
                    (t[1] == '.' || t[1] == ')')) {
                t.erase(0, 2);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                changed = true;
            } else if (t.size() >= 2 && std::isalpha(static_cast<unsigned char>(t[0])) &&
                    (t[1] == '.' || t[1] == ')')) {
                t.erase(0, 2);
                if (!t.empty() && t[0] == ' ') t.erase(0, 1);
                changed = true;
            }
        }

        size_t k = 0;
        while (k < t.size() && !std::isalnum(static_cast<unsigned char>(t[k]))) ++k;
        if (k > 0 && k < t.size()) t.erase(0, k);
    };

    strip_leading_noise(s);
    s = trim(s);

    bool has_alpha = std::any_of(s.begin(), s.end(), [](unsigned char c) {
        return std::isalpha(c);
    });
    if (!has_alpha || s.size() < 12) {
        return "I don't have enough information in my sources.";
    }

    {
        std::string low = to_lower(s);
        const char *fragments[] = {
                "answer only from the provided context",
                "do not repeat the context",
                "respond exactly: \"i don't have enough information in my sources",
                "instructions:",
                "begin your answer",
                "start your response",
                "do not include anything else",
                "reply with only the answer text"
        };
        for (const char *f: fragments) {
            size_t p = low.find(f);
            if (p != std::string::npos) {
                s = trim(s.substr(0, p));
                break;
            }
        }
    }

    return s;
}

// ---------- Chat templating ----------
// Model-agnostic plain prompt format that works with any model.
// This avoids hardcoding model-specific chat templates (like Gemma's <start_of_turn>)
// which cause issues when used with other models (like Llama).
static std::string build_plain_prompt(const std::string &system_msg,
        const std::string &context_block,
        const std::string &user_msg) {
    std::ostringstream oss;

    // Add system instructions if provided
    if (!trim(system_msg).empty()) {
        oss << "Instructions: " << system_msg << "\n\n";
    }

    // Add context if provided
    if (!trim(context_block).empty()) {
        oss << "Context:\n" << context_block << "\n\n";
    }

    // Add user question
    oss << "Question:\n" << user_msg << "\n\nAnswer:\n";
    return oss.str();
}

// ===================================================================================
//                                   EMBEDDINGS
// ===================================================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_llamatik_library_platform_LlamaBridge_initModel(JNIEnv *env, jobject, jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("initModel (embed): %s", path ? path : "(null)");

    if (!g_backend_inited) {
        llama_backend_init();
        g_backend_inited = true;
    }

    llama_model_params mparams = llama_model_default_params();
    // Enable GPU acceleration - offload all layers to GPU (Vulkan on Android)
    mparams.n_gpu_layers = 99;
    LOGI("Loading embed model with n_gpu_layers=%d", mparams.n_gpu_layers);
    emb_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!emb_model) {
        LOGE("embed model load failed");
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.n_ctx = 2048;

    emb_ctx = llama_init_from_model(emb_model, cparams);
    if (!emb_ctx) {
        llama_model_free(emb_model);
        emb_model = nullptr;
        return JNI_FALSE;
    }

    emb_dim = llama_model_n_embd(emb_model);
    LOGI("Embed context ready. dim=%d", emb_dim);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_llamatik_library_platform_LlamaBridge_embed(JNIEnv *env, jobject, jstring input) {
    if (!emb_ctx || !emb_model) {
        LOGE("embed: ctx/model null");
        return nullptr;
    }

    const char *inputStr = env->GetStringUTFChars(input, nullptr);
    if (!inputStr) {
        LOGE("embed: input null");
        return nullptr;
    }

    std::vector<llama_token> tokens(1024);
    int n_tokens = tokenize_with_retry(llama_model_get_vocab(emb_model),
            inputStr, tokens,
            /*add_bos*/ true,
            /*parse_special*/ false);
    env->ReleaseStringUTFChars(input, inputStr);

    if (n_tokens <= 0 || n_tokens > (int)llama_n_ctx(emb_ctx)) {
        LOGW("embed tokenize fail/too long. n=%d ctx=%u", n_tokens, (unsigned)llama_n_ctx(emb_ctx));
        return nullptr;
    }
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }

    if (llama_decode(emb_ctx, batch) != 0) {
        LOGE("embed: llama_decode failed");
        llama_batch_free(batch);
        return nullptr;
    }

    const float *e = llama_get_embeddings_seq(emb_ctx, 0);
    if (!e) {
        LOGE("embed: embeddings null");
        llama_batch_free(batch);
        return nullptr;
    }

    const int dim = llama_model_n_embd(emb_model);
    jfloatArray result = env->NewFloatArray(dim);
    if (!result) {
        llama_batch_free(batch);
        return nullptr;
    }
    env->SetFloatArrayRegion(result, 0, dim, e);
    llama_batch_free(batch);
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_shutdown(JNIEnv *, jobject) {
    LOGI("shutdown: starting, generation_in_progress=%d", g_generation_in_progress.load());

    // Signal cancellation first
    g_cancel_requested.store(true, std::memory_order_release);

    // Wait for any ongoing generation to complete (with timeout)
    // This is critical to prevent use-after-free crashes
    int wait_count = 0;
    const int max_wait_ms = 5000; // 5 second timeout
    const int sleep_ms = 10;
    while (g_generation_in_progress.load(std::memory_order_acquire) && wait_count < (max_wait_ms / sleep_ms)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        wait_count++;
    }

    if (wait_count >= (max_wait_ms / sleep_ms)) {
        LOGE("shutdown: WARNING - generation did not stop within timeout, proceeding anyway");
    } else {
        LOGI("shutdown: generation stopped after %d ms", wait_count * sleep_ms);
    }

    if (emb_ctx) llama_free(emb_ctx);
    if (emb_model) llama_model_free(emb_model);
    emb_ctx = nullptr;
    emb_model = nullptr;

    if (gen_ctx) llama_free(gen_ctx);
    if (gen_model) llama_model_free(gen_model);
    gen_ctx = nullptr;
    gen_model = nullptr;

    if (g_backend_inited) {
        llama_backend_free();
        g_backend_inited = false;
    }

    LOGI("shutdown: complete");
}

// ===================================================================================
//                               TEXT GENERATION
// ===================================================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_llamatik_library_platform_LlamaBridge_initGenerateModel(JNIEnv *env, jobject, jstring modelPath) {
    const char *path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("initGenerateModel: %s", path ? path : "(null)");

    if (!g_backend_inited) {
        // Set up log callback before initializing backend
        llama_log_set(llama_log_callback, nullptr);

        llama_backend_init();
        g_backend_inited = true;

        // Log available backends
        size_t num_devices = ggml_backend_dev_count();
        LOGI("Available backends: %zu devices", num_devices);
        for (size_t i = 0; i < num_devices; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev) {
                const char* name = ggml_backend_dev_name(dev);
                const char* desc = ggml_backend_dev_description(dev);
                enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
                const char* type_str = (type == GGML_BACKEND_DEVICE_TYPE_CPU) ? "CPU" :
                                       (type == GGML_BACKEND_DEVICE_TYPE_GPU) ? "GPU" :
                                       (type == GGML_BACKEND_DEVICE_TYPE_IGPU) ? "IGPU" :
                                       (type == GGML_BACKEND_DEVICE_TYPE_ACCEL) ? "ACCEL" : "UNKNOWN";
                LOGI("  Device %zu: %s (%s) - Type: %s (raw=%d)", i, name ? name : "unknown", desc ? desc : "no desc", type_str, (int)type);
            }
        }
    }

    llama_model_params mparams = llama_model_default_params();
    // Enable GPU acceleration - offload all layers to GPU (Vulkan on Android)
    mparams.n_gpu_layers = 99;
    LOGI("Loading model with n_gpu_layers=%d", mparams.n_gpu_layers);
    gen_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!gen_model) {
        LOGE("gen model load failed");
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = false;
    cparams.n_ctx = 4096;

    gen_ctx = llama_init_from_model(gen_model, cparams);
    if (!gen_ctx) {
        llama_model_free(gen_model);
        gen_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Gen context ready. n_ctx=%u", (unsigned)llama_n_ctx(gen_ctx));
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_llamatik_library_platform_LlamaBridge_generate(JNIEnv *env, jobject, jstring input) {
    if (!gen_ctx || !gen_model) {
        LOGE("generate: ctx/model null");
        return nullptr;
    }

    const char *prompt = env->GetStringUTFChars(input, nullptr);
    if (!prompt) {
        LOGE("generate: prompt null");
        return nullptr;
    }

    llama_memory_clear(llama_get_memory(gen_ctx), false);

    std::vector<llama_token> tokens(2048);
    int n_tokens = tokenize_with_retry(llama_model_get_vocab(gen_model),
            prompt, tokens,
            /*add_bos*/ true,
            /*parse_special*/ true);
    env->ReleaseStringUTFChars(input, prompt);

    if (n_tokens <= 0) {
        LOGE("tokenize failed");
        return nullptr;
    }
    tokens.resize(n_tokens);

    const int n_ctx = (int)llama_n_ctx(gen_ctx);
    if ((int)tokens.size() > n_ctx - 8) truncate_to_ctx(tokens, n_ctx, 8);

    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    batch.n_tokens = (int) tokens.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == batch.n_tokens - 1);
    }
    if (llama_decode(gen_ctx, batch) != 0) {
        llama_batch_free(batch);
        LOGE("decode failed on prompt");
        return nullptr;
    }

    // NOTE: one-shot generate currently uses fixed sampler params (same as before).
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(128, 1.10f, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.80f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.55f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const int max_new_tokens = 640;
    int cur_pos = batch.n_tokens;

    std::string output;
    char buf[8192];

    for (int i = 0; i < max_new_tokens; ++i) {
        llama_token tok = llama_sampler_sample(sampler, gen_ctx, -1);
        if (tok < 0) break;
        if (tok == llama_vocab_eos(llama_model_get_vocab(gen_model))) break;

        // early stop on chat EOT
        char sp[64];
        int sn = llama_token_to_piece(llama_model_get_vocab(gen_model), tok, sp, (int) sizeof(sp), 0, 1);
        if (sn > 0) {
            sp[std::min(sn, (int) sizeof(sp) - 1)] = '\0';
            if (std::strcmp(sp, "<end_of_turn>") == 0 || std::strcmp(sp, "<|eot_id|>") == 0) break;
        }

        llama_sampler_accept(sampler, tok);

        int nn = llama_token_to_piece(llama_model_get_vocab(gen_model), tok, buf, (int) sizeof(buf), 0, 0);
        if (nn > 0) output.append(buf, nn);

        if (cur_pos >= n_ctx) break;

        llama_batch step = llama_batch_init(1, 0, 1);
        step.n_tokens = 1;
        step.token[0] = tok;
        step.pos[0] = cur_pos++;
        step.n_seq_id[0] = 1;
        step.seq_id[0][0] = 0;
        step.logits[0] = true;

        if (llama_decode(gen_ctx, step) != 0) {
            llama_batch_free(step);
            break;
        }
        llama_batch_free(step);
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    std::string clean = sanitize_generation(output);
    return env->NewStringUTF(clean.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_llamatik_library_platform_LlamaBridge_generateWithContext(
        JNIEnv *env, jobject, jstring jSystem, jstring jContext, jstring jUser) {

    const char *psys = jSystem ? env->GetStringUTFChars(jSystem, nullptr) : nullptr;
    const char *pctx = jContext ? env->GetStringUTFChars(jContext, nullptr) : nullptr;
    const char *pusr = env->GetStringUTFChars(jUser, nullptr);

    std::string system = psys ? psys : "";
    std::string ctx = pctx ? pctx : "";
    std::string user = pusr ? pusr : "";

    if (jSystem) env->ReleaseStringUTFChars(jSystem, psys);
    if (jContext) env->ReleaseStringUTFChars(jContext, pctx);
    if (jUser) env->ReleaseStringUTFChars(jUser, pusr);

    // Use model-agnostic plain prompt format
    std::string prompt = build_plain_prompt(system, ctx, user);
    jstring jp = env->NewStringUTF(prompt.c_str());
    jstring r = Java_com_llamatik_library_platform_LlamaBridge_generate(env, nullptr, jp);
    env->DeleteLocalRef(jp);
    return r;
}

// ===================================================================================
//                        REAL TOKEN STREAMING (JNI CALLBACKS)
// ===================================================================================

struct StreamMethods {
    jmethodID onDelta;
    jmethodID onComplete;
    jmethodID onError;
};

static bool resolve_stream_methods(JNIEnv *env, jobject cb, StreamMethods &m) {
    jclass cls = env->GetObjectClass(cb);
    if (!cls) return false;
    m.onDelta = env->GetMethodID(cls, "onDelta", "(Ljava/lang/String;)V");
    m.onComplete = env->GetMethodID(cls, "onComplete", "()V");
    m.onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
    return m.onDelta && m.onComplete && m.onError;
}

static inline bool is_eot_piece(const char *s) {
    return std::strcmp(s, "<end_of_turn>") == 0 || std::strcmp(s, "<|eot_id|>") == 0;
}

// Streams tokens from a prepared prompt string
static void stream_from_prompt(JNIEnv *env, const char *prompt, jobject jCallback, const StreamMethods &m) {
    // Serialize generation calls to prevent KV cache corruption
    // This ensures only one generation runs at a time
    std::lock_guard<std::mutex> lock(g_generation_mutex);

    if (!gen_ctx || !gen_model) {
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("model not initialized"));
        return;
    }

    // Mark generation as in progress (for safe shutdown)
    g_generation_in_progress.store(true, std::memory_order_release);

    // Reset cancel flag at the start of each stream
    g_cancel_requested.store(false, std::memory_order_relaxed);
    llama_memory_clear(llama_get_memory(gen_ctx), false);

    std::vector<llama_token> tokens(2048);
    int n_tokens = tokenize_with_retry(llama_model_get_vocab(gen_model),
            prompt, tokens,
            /*add_bos*/ true,
            /*parse_special*/ true);
    if (n_tokens <= 0) {
        g_generation_in_progress.store(false, std::memory_order_release);
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("tokenization failed"));
        return;
    }
    tokens.resize(n_tokens);

    const int n_ctx = (int)llama_n_ctx(gen_ctx);
    if ((int)tokens.size() > n_ctx - 8) truncate_to_ctx(tokens, n_ctx, 8);

    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    batch.n_tokens = (int) tokens.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == batch.n_tokens - 1);
    }
    if (llama_decode(gen_ctx, batch) != 0) {
        llama_batch_free(batch);
        g_generation_in_progress.store(false, std::memory_order_release);
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("llama_decode failed on prompt"));
        return;
    }

    float temperature    = g_temperature.load();
    float top_p          = g_top_p.load();
    int   top_k          = g_top_k.load();
    float repeat_penalty = g_repeat_penalty.load();
    int   max_new_tokens = g_max_new_tokens.load();

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    int cur_pos = batch.n_tokens;

    char piece_buf[768];
    char spec_buf[64];

    for (int i = 0; i < max_new_tokens; ++i) {
        if (g_cancel_requested.load(std::memory_order_relaxed)) {
            break;
        }

        llama_token tok = llama_sampler_sample(sampler, gen_ctx, -1);
        if (tok < 0) break;
        if (tok == llama_vocab_eos(llama_model_get_vocab(gen_model))) break;

        int sn = llama_token_to_piece(llama_model_get_vocab(gen_model),
                tok, spec_buf, (int) sizeof(spec_buf),
                /* lstrip */ 0, /* special */ 1);
        if (sn > 0) {
            spec_buf[std::min(sn, (int) sizeof(spec_buf) - 1)] = '\0';
            if (is_eot_piece(spec_buf) || std::strcmp(spec_buf, "<start_of_turn>") == 0) {
                break;
            }
        }

        llama_sampler_accept(sampler, tok);

        int nn = llama_token_to_piece(llama_model_get_vocab(gen_model),
                tok, piece_buf, (int) sizeof(piece_buf),
                /* lstrip */ 0, /* special */ 0);
        if (nn > 0) {
            piece_buf[std::min(nn, (int) sizeof(piece_buf) - 1)] = '\0';
            jstring delta = env->NewStringUTF(piece_buf);
            if (delta) {
                env->CallVoidMethod(jCallback, m.onDelta, delta);
                env->DeleteLocalRef(delta);
            }
        }

        if (cur_pos >= n_ctx) break;

        llama_batch step = llama_batch_init(1, 0, 1);
        step.n_tokens = 1;
        step.token[0] = tok;
        step.pos[0] = cur_pos++;
        step.n_seq_id[0] = 1;
        step.seq_id[0][0] = 0;
        step.logits[0] = true;

        if (llama_decode(gen_ctx, step) != 0) {
            llama_batch_free(step);
            llama_sampler_free(sampler);
            llama_batch_free(batch);
            g_generation_in_progress.store(false, std::memory_order_release);
            env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("llama_decode failed mid-stream"));
            return;
        }
        llama_batch_free(step);
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    // Mark generation as complete (for safe shutdown)
    g_generation_in_progress.store(false, std::memory_order_release);

    // Always signal completion – Kotlin side will ignore if it has nulled activeRequestId
    env->CallVoidMethod(jCallback, m.onComplete);
}

// JNI: stream(prompt, callback)
extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeGenerateStream(
        JNIEnv *env, jobject /*thiz*/, jstring jPrompt, jobject jCallback) {
    if (!jPrompt || !jCallback) return;

    StreamMethods m{};
    if (!resolve_stream_methods(env, jCallback, m)) {
        LOGE("nativeGenerateStream: failed to resolve callback methods");
        return;
    }

    const char *prompt = env->GetStringUTFChars(jPrompt, nullptr);
    if (!prompt) {
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("prompt decode failed"));
        return;
    }

    stream_from_prompt(env, prompt, jCallback, m);
    env->ReleaseStringUTFChars(jPrompt, prompt);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeCancelGenerate(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    LOGI("nativeCancelGenerate: cancel requested");
    g_cancel_requested.store(true, std::memory_order_relaxed);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeGenerateWithContextStream(
        JNIEnv *env, jobject /*thiz*/,
        jstring jSystem, jstring jContext, jstring jUser, jobject jCallback) {
    if (!jCallback) return;

    StreamMethods m{};
    if (!resolve_stream_methods(env, jCallback, m)) {
        LOGE("nativeGenerateWithContextStream: failed to resolve callback methods");
        return;
    }

    const char *psys = jSystem ? env->GetStringUTFChars(jSystem, nullptr) : nullptr;
    const char *pctx = jContext ? env->GetStringUTFChars(jContext, nullptr) : nullptr;
    const char *pusr = jUser ? env->GetStringUTFChars(jUser, nullptr) : nullptr;

    std::string system = psys ? psys : "";
    std::string ctx = pctx ? pctx : "";
    std::string user = pusr ? pusr : "";

    if (jSystem) env->ReleaseStringUTFChars(jSystem, psys);
    if (jContext) env->ReleaseStringUTFChars(jContext, pctx);
    if (jUser) env->ReleaseStringUTFChars(jUser, pusr);

    // Use model-agnostic plain prompt format
    std::string prompt = build_plain_prompt(system, ctx, user);

    stream_from_prompt(env, prompt.c_str(), jCallback, m);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeUpdateGenerationParams(
        JNIEnv * /*env*/,
        jobject /*thiz*/,
        jfloat temperature,
        jint maxTokens,
        jfloat topP,
        jint topK,
        jfloat repeatPenalty) {
    g_temperature     = temperature;
    g_top_p           = topP;
    g_top_k           = topK;
    g_repeat_penalty  = repeatPenalty;
    g_max_new_tokens  = (int)maxTokens;
}

// ===================================================================================
//                    CHAT TEMPLATE STREAMING (Message Array API)
// ===================================================================================

/**
 * Build a prompt using llama.cpp's chat template system.
 * Uses the model's embedded template if available, otherwise falls back to ChatML.
 *
 * @param messages Vector of {role, content} pairs
 * @return Formatted prompt string ready for tokenization
 */
static std::string build_chat_template_prompt(const std::vector<llama_chat_message> &messages) {
    if (!gen_model) {
        LOGE("build_chat_template_prompt: model not loaded");
        return "";
    }

    // Get the model's embedded chat template (may be nullptr)
    const char *model_template = llama_model_chat_template(gen_model, nullptr);

    if (model_template) {
        LOGI("Using model's embedded chat template");
    } else {
        LOGI("Model has no embedded template, using ChatML fallback");
    }

    // Estimate buffer size: 2x total content length + overhead for template tokens
    size_t total_content_len = 0;
    for (const auto &msg : messages) {
        total_content_len += std::strlen(msg.role) + std::strlen(msg.content);
    }
    size_t buf_size = std::max(total_content_len * 3 + 256, (size_t)4096);

    std::vector<char> buf(buf_size);

    // Apply the chat template
    // If model_template is nullptr, llama_chat_apply_template will use ChatML as default
    int32_t result = llama_chat_apply_template(
            model_template,
            messages.data(),
            messages.size(),
            true,  // add_ass: add assistant turn start tokens
            buf.data(),
            (int32_t)buf.size()
    );

    if (result < 0) {
        LOGE("llama_chat_apply_template failed with error %d", result);
        return "";
    }

    // If buffer was too small, resize and retry
    if ((size_t)result > buf.size()) {
        buf.resize(result + 1);
        result = llama_chat_apply_template(
                model_template,
                messages.data(),
                messages.size(),
                true,
                buf.data(),
                (int32_t)buf.size()
        );
        if (result < 0) {
            LOGE("llama_chat_apply_template retry failed with error %d", result);
            return "";
        }
    }

    std::string prompt(buf.data(), result);
    LOGD("Chat template prompt (%d chars): %.100s...", result, prompt.c_str());
    return prompt;
}

/**
 * JNI: Stream generation using chat template with message array.
 *
 * @param jRoles Array of role strings ("system", "user", "assistant")
 * @param jContents Array of content strings (message text)
 * @param jCallback GenStream callback object
 */
extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeGenerateWithMessagesStream(
        JNIEnv *env, jobject /*thiz*/,
        jobjectArray jRoles, jobjectArray jContents, jobject jCallback) {
    if (!jCallback || !jRoles || !jContents) {
        LOGE("nativeGenerateWithMessagesStream: null arguments");
        return;
    }

    StreamMethods m{};
    if (!resolve_stream_methods(env, jCallback, m)) {
        LOGE("nativeGenerateWithMessagesStream: failed to resolve callback methods");
        return;
    }

    jsize n_messages = env->GetArrayLength(jRoles);
    jsize n_contents = env->GetArrayLength(jContents);

    if (n_messages != n_contents) {
        env->CallVoidMethod(jCallback, m.onError,
                env->NewStringUTF("roles and contents arrays must have same length"));
        return;
    }

    if (n_messages == 0) {
        env->CallVoidMethod(jCallback, m.onError,
                env->NewStringUTF("messages array is empty"));
        return;
    }

    // Build the messages vector
    std::vector<llama_chat_message> messages;
    std::vector<std::string> role_strings;    // Keep strings alive
    std::vector<std::string> content_strings;
    messages.reserve(n_messages);
    role_strings.reserve(n_messages);
    content_strings.reserve(n_messages);

    for (jsize i = 0; i < n_messages; i++) {
        jstring jRole = (jstring)env->GetObjectArrayElement(jRoles, i);
        jstring jContent = (jstring)env->GetObjectArrayElement(jContents, i);

        const char *role_cstr = jRole ? env->GetStringUTFChars(jRole, nullptr) : nullptr;
        const char *content_cstr = jContent ? env->GetStringUTFChars(jContent, nullptr) : nullptr;

        role_strings.emplace_back(role_cstr ? role_cstr : "user");
        content_strings.emplace_back(content_cstr ? content_cstr : "");

        if (jRole && role_cstr) env->ReleaseStringUTFChars(jRole, role_cstr);
        if (jContent && content_cstr) env->ReleaseStringUTFChars(jContent, content_cstr);

        messages.push_back({role_strings.back().c_str(), content_strings.back().c_str()});
    }

    LOGI("nativeGenerateWithMessagesStream: %d messages", (int)messages.size());

    // Build prompt using chat template
    std::string prompt = build_chat_template_prompt(messages);

    if (prompt.empty()) {
        env->CallVoidMethod(jCallback, m.onError,
                env->NewStringUTF("failed to build chat template prompt"));
        return;
    }

    // Stream from the templated prompt
    stream_from_prompt(env, prompt.c_str(), jCallback, m);
}
