#include <jni.h>
#include "llama.h"
#include "llama_jni.h"
#include "engine_state.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

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
#include <memory>
#include <utility>

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
static struct mtmd_context *vision_ctx = nullptr;


// Backend lifetime
static bool g_backend_inited = false;

// Captures the last error message from llama.cpp for surfacing to the user
static std::string g_last_error;
static std::mutex g_last_error_mutex;

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
        case GGML_LOG_LEVEL_ERROR: {
            LOGE("llama: %s", msg.c_str());
            std::lock_guard<std::mutex> lock(g_last_error_mutex);
            g_last_error = msg;
            break;
        }
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

// Sessions, cancellation, the generation lock and the resident model path live in
// engine_state, shared with the iOS bridge. Keeping private copies is what let the same
// three bugs be fixed on one platform and left on the other.
using llamatik::GenerationLock;
using llamatik::TeardownLock;
using llamatik::should_cancel_session;

static std::atomic<float> g_temperature     = 0.7f;   // align with app default
static std::atomic<float> g_top_p          = 0.95f;  // align with app default
static std::atomic<int>   g_top_k          = 40;     // align with app default
static std::atomic<float> g_repeat_penalty = 1.10f;
static std::atomic<int>   g_max_new_tokens = 512;    // align with app default

// ===================================================================================
//                              SMALL HELPERS
// ===================================================================================


// Carries the session through llama.cpp's abort callback, which only takes a void*.
struct vision_abort_state {
    uint64_t session_id;
};

// Returning true tells ggml to stop the graph mid-compute. Called frequently from the compute
// threads, so it must stay a relaxed atomic read and nothing more.
static bool vision_should_abort(void *data) {
    const auto *state = static_cast<const vision_abort_state *>(data);
    return state != nullptr && should_cancel_session(state->session_id);
}

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
    LOGI("shutdown: starting, generation_in_progress=%d", llamatik::generation_in_progress());

    // Cancels every session and holds the generation lock until this scope ends, so
    // nothing can start generating on a context that is about to be freed.
    TeardownLock teardown;

    LOGI("shutdown: mutex acquired, freeing resources");

    if (emb_ctx) llama_free(emb_ctx);
    if (emb_model) llama_model_free(emb_model);
    emb_ctx = nullptr;
    emb_model = nullptr;

    if (gen_ctx) llama_free(gen_ctx);
    if (gen_model) llama_model_free(gen_model);
    if (vision_ctx) mtmd_free(vision_ctx);
    gen_ctx = nullptr;
    gen_model = nullptr;
    vision_ctx = nullptr;
    llamatik::clear_resident_model_path();

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
        // ggml logs separately from llama, and ggml_abort() writes its assertion text
        // through this callback before dying. Without it, a GGML_ASSERT failure reaches
        // Android as a bare SIGABRT with the message lost on stderr.
        ggml_log_set(llama_log_callback, nullptr);

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

    // Cancel and hold the generation lock across the teardown *and* the load below.
    // The comment here used to claim this prevented a use-after-free; it did not, because
    // nothing stopped a generation running while these frees happened. On iOS, which has
    // no Kotlin layer draining generation first, that is exactly what crashed.
    TeardownLock teardown;

    if (vision_ctx) {
        LOGW("initGenerateModel: freeing existing vision_ctx before re-init");
        mtmd_free(vision_ctx);
        vision_ctx = nullptr;
    }
    if (gen_ctx) {
        LOGW("initGenerateModel: freeing existing gen_ctx before re-init");
        llama_free(gen_ctx);
        gen_ctx = nullptr;
    }
    if (gen_model) {
        LOGW("initGenerateModel: freeing existing gen_model before re-init");
        llama_model_free(gen_model);
        gen_model = nullptr;
    }
    // This path owns gen_model too, so it owns the record of what gen_model is.
    llamatik::clear_resident_model_path();

    // Clear last error before loading
    {
        std::lock_guard<std::mutex> lock(g_last_error_mutex);
        g_last_error.clear();
    }

    llama_model_params mparams = llama_model_default_params();
    // Enable GPU acceleration - offload all layers to GPU (Vulkan on Android)
    mparams.n_gpu_layers = 99;
    LOGI("Loading model with n_gpu_layers=%d", mparams.n_gpu_layers);
    gen_model = llama_model_load_from_file(path, mparams);
    // Copy before handing the chars back to the JVM; `path` dangles after this.
    const std::string requested_path = path ? path : "";
    env->ReleaseStringUTFChars(modelPath, path);

    if (!gen_model) {
        LOGE("gen model load failed");
        std::string error;
        {
            std::lock_guard<std::mutex> lock(g_last_error_mutex);
            error = g_last_error;
        }
        if (error.empty()) error = "Failed to load model";
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, error.c_str());
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = false;
    cparams.n_ctx = 4096;

    gen_ctx = llama_init_from_model(gen_model, cparams);
    if (!gen_ctx) {
        llama_model_free(gen_model);
        gen_model = nullptr;
        llamatik::clear_resident_model_path();
        jclass exClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exClass, "Failed to create model context");
        return JNI_FALSE;
    }

    llamatik::set_resident_model_path(requested_path);

    LOGI("Gen context ready. n_ctx=%u", (unsigned)llama_n_ctx(gen_ctx));
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_llamatik_library_platform_LlamaBridge_generate(JNIEnv *env, jobject, jstring input) {
    LOGI("generate (non-streaming): starting");

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

    // Read params from atomics (set by updateGenerateParams from Kotlin)
    float temperature    = g_temperature.load();
    float top_p          = g_top_p.load();
    int   top_k          = g_top_k.load();
    float repeat_penalty = g_repeat_penalty.load();
    int   max_new_tokens = g_max_new_tokens.load();

    LOGI("generate (non-streaming): params temp=%.2f top_p=%.2f top_k=%d max_tokens=%d",
         temperature, top_p, top_k, max_new_tokens);

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(llama_vocab_n_tokens(llama_model_get_vocab(gen_model)), 128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
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

    LOGI("generate (non-streaming): complete");

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

// Number of bytes in the UTF-8 sequence that starts with `lead`, or 0 if
// `lead` is a continuation byte / not a valid sequence start.
static inline int utf8_seq_len(unsigned char lead) {
    if ((lead & 0x80) == 0x00) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 0;
}

// Token pieces from llama_token_to_piece can split a multi-byte UTF-8
// character (e.g. an emoji) across a token boundary. JNI's NewStringUTF
// aborts the whole process if handed a partial sequence, so streaming
// callbacks must buffer bytes until a full character is available.
// Appends `data` to `pending`, returns the longest valid-UTF8 prefix ready
// to emit, and leaves any trailing incomplete sequence in `pending`.
static std::string utf8_append_and_flush(std::string &pending, const char *data, int n) {
    pending.append(data, n);
    size_t flush_len = pending.size();
    for (size_t back = 1; back <= 4 && back <= pending.size(); ++back) {
        unsigned char b = (unsigned char) pending[pending.size() - back];
        int len = utf8_seq_len(b);
        if (len == 0) continue; // continuation byte, keep scanning backward
        if ((int) back < len) flush_len = pending.size() - back;
        break;
    }
    std::string ready = pending.substr(0, flush_len);
    pending.erase(0, flush_len);
    return ready;
}

// Streams tokens from a prepared prompt string
static void stream_from_prompt(JNIEnv *env, const char *prompt, jobject jCallback, const StreamMethods &m) {
    LOGI("stream_from_prompt: waiting for mutex...");

    // Serialize generation calls to prevent KV cache corruption
    // This ensures only one generation runs at a time
    // Holds the generation lock and marks generation in progress until this returns,
    // by any path. The manual flag stores this replaces were missing from some of them.
    GenerationLock lock;

    // After the lock, so session IDs are unique. Also drops a cancellation left over from
    // an earlier session, which would otherwise stop this one before its first token.
    const uint64_t session_id = llamatik::begin_session();
    LOGI("stream_from_prompt: lock acquired, session_id=%llu, cancel_session=%llu",
         (unsigned long long)session_id,
         (unsigned long long)llamatik::pending_cancel_session());

    if (!gen_ctx || !gen_model) {
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("model not initialized"));
        return;
    }


    // Clear KV cache completely (data=true) to ensure consistent prompt processing speed
    llama_memory_clear(llama_get_memory(gen_ctx), true);

    std::vector<llama_token> tokens(2048);
    int n_tokens = tokenize_with_retry(llama_model_get_vocab(gen_model),
            prompt, tokens,
            /*add_bos*/ true,
            /*parse_special*/ true);
    if (n_tokens <= 0) {
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
    LOGI("stream_from_prompt: decoding prompt (%d tokens)...", batch.n_tokens);
    if (llama_decode(gen_ctx, batch) != 0) {
        llama_batch_free(batch);
        env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("llama_decode failed on prompt"));
        return;
    }
    LOGI("stream_from_prompt: prompt decoded successfully");

    float temperature    = g_temperature.load();
    float top_p          = g_top_p.load();
    int   top_k          = g_top_k.load();
    float repeat_penalty = g_repeat_penalty.load();
    int   max_new_tokens = g_max_new_tokens.load();

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(llama_vocab_n_tokens(llama_model_get_vocab(gen_model)), 128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    int cur_pos = batch.n_tokens;

    char piece_buf[768];
    char spec_buf[64];
    std::string pending_utf8;

    LOGI("stream_from_prompt: starting token generation loop, max_tokens=%d, session=%llu",
         max_new_tokens, (unsigned long long)session_id);

    for (int i = 0; i < max_new_tokens; ++i) {
        // Check if THIS session should be cancelled (not a different one). The pending
        // ID is only read for the log, so it stays out of the per-token path.
        if (should_cancel_session(session_id)) {
            LOGI("stream_from_prompt: cancel requested for session %llu at token %d (cancel_id=%llu)",
                 (unsigned long long)session_id, i,
                 (unsigned long long)llamatik::pending_cancel_session());
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
            std::string ready = utf8_append_and_flush(pending_utf8, piece_buf, nn);
            if (!ready.empty()) {
                jstring delta = env->NewStringUTF(ready.c_str());
                if (delta) {
                    env->CallVoidMethod(jCallback, m.onDelta, delta);
                    env->DeleteLocalRef(delta);
                }
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
                env->CallVoidMethod(jCallback, m.onError, env->NewStringUTF("llama_decode failed mid-stream"));
            return;
        }
        llama_batch_free(step);
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    // Mark generation as complete (for safe shutdown)

    LOGI("stream_from_prompt: generation complete for session %llu, releasing mutex",
         (unsigned long long)session_id);

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
    // Cancel the CURRENT session only
    // This prevents race conditions where a late cancellation affects a new generation
    uint64_t current_session = llamatik::current_session();
    uint64_t old_cancel = llamatik::pending_cancel_session();
    LOGI("nativeCancelGenerate: requesting cancel for session %llu (was %llu, in_progress=%d)",
         (unsigned long long)current_session,
         (unsigned long long)old_cancel,
         llamatik::generation_in_progress() ? 1 : 0);
    llamatik::cancel_session(current_session);
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

    char arch_buf[64] = {};
    const int32_t arch_len = llama_model_meta_val_str(
            gen_model,
            "general.architecture",
            arch_buf,
            (int32_t)sizeof(arch_buf)
    );
    const std::string architecture = arch_len > 0 ? std::string(arch_buf) : "";
    const bool is_gemma_family = architecture.rfind("gemma", 0) == 0;

    // Estimate buffer size: 2x total content length + overhead for template tokens
    size_t total_content_len = 0;
    for (const auto &msg : messages) {
        total_content_len += std::strlen(msg.role) + std::strlen(msg.content);
    }
    size_t buf_size = std::max(total_content_len * 3 + 256, (size_t)4096);

    std::vector<char> buf(buf_size);

    auto apply_template = [&](const char *tmpl, const char *label) -> int32_t {
        int32_t apply_result = llama_chat_apply_template(
                tmpl,
                messages.data(),
                messages.size(),
                true,  // add_ass: add assistant turn start tokens
                buf.data(),
                (int32_t)buf.size()
        );

        if (apply_result < 0) {
            LOGW("llama_chat_apply_template failed for %s with error %d", label, apply_result);
            return apply_result;
        }

        if ((size_t)apply_result > buf.size()) {
            buf.resize(apply_result + 1);
            apply_result = llama_chat_apply_template(
                    tmpl,
                    messages.data(),
                    messages.size(),
                    true,
                    buf.data(),
                    (int32_t)buf.size()
            );
            if (apply_result < 0) {
                LOGW("llama_chat_apply_template retry failed for %s with error %d", label, apply_result);
            }
        }

        return apply_result;
    };

    // If model_template is nullptr, llama_chat_apply_template will use ChatML as default.
    int32_t result = apply_template(model_template, model_template ? "embedded template" : "ChatML fallback");

    if (result < 0 && model_template && is_gemma_family) {
        LOGW("Embedded chat template is not supported for %s; retrying with Gemma template", architecture.c_str());
        result = apply_template("gemma", "Gemma fallback");
    }

    if (result < 0 && model_template && !is_gemma_family) {
        LOGW("Embedded chat template is not supported for %s; retrying with ChatML fallback", architecture.c_str());
        result = apply_template(nullptr, "ChatML fallback");
    }

    if (result < 0) {
        LOGE("Unable to format chat prompt with embedded/default/fallback templates");
        return "";
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

extern "C"
bool llama_vision_available(void) {
    return true;
}

extern "C"
bool llama_vision_init(const char *model_path, const char *projection_model_path) {
    if (!model_path || !projection_model_path) {
        LOGE("llama_vision_init: missing model or projection path");
        return false;
    }

    if (!g_backend_inited) {
        llama_log_set(llama_log_callback, nullptr);
        // ggml logs separately from llama, and ggml_abort() writes its assertion text
        // through this callback before dying. Without it, a GGML_ASSERT failure reaches
        // Android as a bare SIGABRT with the message lost on stderr.
        ggml_log_set(llama_log_callback, nullptr);
        llama_backend_init();
        g_backend_inited = true;
    }

    // Reuse the resident model only when it is the one being asked for. initGenerateModel()
    // already frees unconditionally before it loads; this is the same rule, stated as a match
    // so that re-selecting the current vision model does not reload gigabytes for nothing.
    if ((gen_model || gen_ctx) && !llamatik::can_reuse_resident_model(true, model_path)) {
        LOGI("llama_vision_init: replacing resident model '%s' with '%s'",
             llamatik::resident_model_path().c_str(), model_path);
        // Nothing may be generating into the context while it is freed.
        TeardownLock teardown;
        if (vision_ctx) {
            mtmd_free(vision_ctx);
            vision_ctx = nullptr;
        }
        if (gen_ctx) {
            llama_free(gen_ctx);
            gen_ctx = nullptr;
        }
        if (gen_model) {
            llama_model_free(gen_model);
            gen_model = nullptr;
        }
        llamatik::clear_resident_model_path();
    }

    if (!gen_model || !gen_ctx) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 99;
        gen_model = llama_model_load_from_file(model_path, mparams);
        if (!gen_model) {
            LOGE("llama_vision_init: failed to load base model");
            return false;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.embeddings = false;
        cparams.n_ctx = 4096;
        gen_ctx = llama_init_from_model(gen_model, cparams);
        if (!gen_ctx) {
            LOGE("llama_vision_init: failed to create base model context");
            llama_model_free(gen_model);
            gen_model = nullptr;
            return false;
        }
        llamatik::set_resident_model_path(model_path);
    }

    if (vision_ctx) {
        mtmd_free(vision_ctx);
        vision_ctx = nullptr;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = true;
    mparams.print_timings = false;
    mparams.n_threads = std::max(1u, std::thread::hardware_concurrency());
    vision_ctx = mtmd_init_from_file(projection_model_path, gen_model, mparams);
    if (!vision_ctx) {
        LOGE("llama_vision_init: failed to load projector");
        return false;
    }

    if (!mtmd_support_vision(vision_ctx)) {
        LOGE("llama_vision_init: projector does not report vision support");
        mtmd_free(vision_ctx);
        vision_ctx = nullptr;
        return false;
    }

    return true;
}

static void vision_stream_from_pos(
        uint64_t session_id,
        llama_pos start_pos,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user) {
    const float temperature = g_temperature.load();
    const float top_p = g_top_p.load();
    const int top_k = g_top_k.load();
    const float repeat_penalty = g_repeat_penalty.load();
    const int max_new_tokens = g_max_new_tokens.load();

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) {
        if (on_error) on_error("sampler init failed", user);
        return;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(llama_vocab_n_tokens(llama_model_get_vocab(gen_model)), 128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_vocab *vocab = llama_model_get_vocab(gen_model);
    const int n_ctx = (int)llama_n_ctx(gen_ctx);
    int cur_pos = (int)start_pos;
    char piece_buf[768];
    char spec_buf[64];
    std::string pending_utf8;

    for (int i = 0; i < max_new_tokens; ++i) {
        if (should_cancel_session(session_id)) break;

        llama_token tok = llama_sampler_sample(sampler, gen_ctx, -1);
        if (tok < 0 || llama_vocab_is_eog(vocab, tok)) break;

        int sn = llama_token_to_piece(vocab, tok, spec_buf, (int)sizeof(spec_buf), 0, 1);
        if (sn > 0) {
            spec_buf[std::min(sn, (int)sizeof(spec_buf) - 1)] = '\0';
            if (is_eot_piece(spec_buf) || std::strcmp(spec_buf, "<start_of_turn>") == 0) break;
        }

        llama_sampler_accept(sampler, tok);

        int nn = llama_token_to_piece(vocab, tok, piece_buf, (int)sizeof(piece_buf), 0, 0);
        if (nn > 0) {
            std::string ready = utf8_append_and_flush(pending_utf8, piece_buf, nn);
            if (!ready.empty() && on_delta) {
                on_delta(ready.c_str(), user);
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
            if (on_error) on_error("llama_decode failed mid-stream", user);
            return;
        }
        llama_batch_free(step);
    }

    llama_sampler_free(sampler);
    if (on_done) on_done(user);
}

static std::string build_vision_prompt(const char **roles, const char **contents, int n_messages, int n_images) {
    std::vector<std::string> role_strings;
    std::vector<std::string> content_strings;
    std::vector<llama_chat_message> messages;
    role_strings.reserve(n_messages);
    content_strings.reserve(n_messages);
    messages.reserve(n_messages);

    int marker_target = -1;
    for (int i = n_messages - 1; i >= 0; --i) {
        if (roles[i] && std::strcmp(roles[i], "user") == 0) {
            marker_target = i;
            break;
        }
    }
    if (marker_target < 0 && n_messages > 0) marker_target = n_messages - 1;

    const char *marker = mtmd_get_marker(vision_ctx);
    if (!marker) marker = mtmd_default_marker();

    for (int i = 0; i < n_messages; ++i) {
        role_strings.emplace_back(roles[i] ? roles[i] : "user");
        content_strings.emplace_back(contents[i] ? contents[i] : "");
        if (i == marker_target) {
            for (int image_index = 0; image_index < n_images; ++image_index) {
                if (!content_strings.back().empty()) content_strings.back().append("\n");
                content_strings.back().append(marker);
            }
        }
        messages.push_back({role_strings.back().c_str(), content_strings.back().c_str()});
    }

    return build_chat_template_prompt(messages);
}

extern "C"
void llama_vision_generate_messages_stream(const char **roles,
        const char **contents,
        int n_messages,
        const char **image_paths,
        int n_images,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user) {
    GenerationLock lock;
    const uint64_t session_id = llamatik::begin_session();

    if (!gen_ctx || !gen_model || !vision_ctx) {
        if (on_error) on_error("vision model not initialized", user);
        return;
    }
    if (!roles || !contents || n_messages <= 0) {
        if (on_error) on_error("invalid arguments: roles, contents, or n_messages", user);
        return;
    }
    if (n_images < 0 || (n_images > 0 && !image_paths)) {
        if (on_error) on_error("invalid image arguments", user);
        return;
    }

    llama_memory_clear(llama_get_memory(gen_ctx), true);

    std::vector<mtmd_bitmap *> bitmaps;
    std::vector<const mtmd_bitmap *> bitmap_refs;
    bitmaps.reserve(n_images);
    bitmap_refs.reserve(n_images);
    for (int i = 0; i < n_images; ++i) {
        auto bitmap = mtmd_helper_bitmap_init_from_file(vision_ctx, image_paths[i], false, mtmd_helper_init_opt_default());
        if (!bitmap.bitmap) {
            for (mtmd_bitmap *loaded : bitmaps) mtmd_bitmap_free(loaded);
                if (on_error) on_error("failed to load image", user);
            return;
        }
        bitmaps.push_back(bitmap.bitmap);
        bitmap_refs.push_back(bitmap.bitmap);
        if (bitmap.video_ctx) {
            mtmd_helper_video_free(bitmap.video_ctx);
        }
    }

    std::string prompt = build_vision_prompt(roles, contents, n_messages, n_images);
    if (prompt.empty()) {
        for (mtmd_bitmap *bitmap : bitmaps) mtmd_bitmap_free(bitmap);
        if (on_error) on_error("failed to build chat template prompt", user);
        return;
    }

    mtmd_input_text text{};
    text.text = prompt.c_str();
    // b10809 added text_len; mtmd now reads exactly this many bytes rather than to the
    // terminator. Leaving it zero tokenizes an empty prompt, which yields no chunks, decodes
    // nothing, and only surfaces later as a null-logits abort inside the sampler.
    text.text_len = prompt.size();
    text.add_special = true;
    text.parse_special = true;

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    int32_t tokenize_result = mtmd_tokenize(
            vision_ctx,
            chunks,
            &text,
            bitmap_refs.empty() ? nullptr : bitmap_refs.data(),
            bitmap_refs.size());

    for (mtmd_bitmap *bitmap : bitmaps) mtmd_bitmap_free(bitmap);

    if (tokenize_result != 0) {
        mtmd_input_chunks_free(chunks);
        if (on_error) on_error("failed to tokenize multimodal prompt", user);
        return;
    }

    // Evaluate the prompt one chunk at a time rather than through mtmd_helper_eval_chunks(),
    // so cancellation is checked between chunks. The helper evaluates everything in a single
    // call, which meant shutdown() could not get the generation mutex until every image had
    // been encoded — on a large vision model that is many seconds of uninterruptible work, and
    // it froze whichever thread was waiting to free the model.
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    llama_pos new_n_past = 0;
    int32_t eval_result = 0;
    bool cancelled = false;

    // Also abort inside llama_decode(), which handles the text half of the prompt and the
    // per-image embedding decode. The image encode itself stays uninterruptible: clip exposes
    // no abort hook, so a single image still runs to completion.
    vision_abort_state abort_state{session_id};
    llama_set_abort_callback(gen_ctx, vision_should_abort, &abort_state);

    for (size_t i = 0; i < n_chunks; ++i) {
        if (should_cancel_session(session_id)) {
            LOGI("vision_generate: cancel requested for session %llu at chunk %zu/%zu",
                 (unsigned long long)session_id, i, n_chunks);
            cancelled = true;
            break;
        }
        const bool is_last = (i + 1 == n_chunks);
        eval_result = mtmd_helper_eval_chunk_single(
                vision_ctx,
                gen_ctx,
                mtmd_input_chunks_get(chunks, i),
                new_n_past,
                0,
                512,
                /* logits_last */ is_last,
                &new_n_past);
        if (eval_result != 0) break;
    }

    llama_set_abort_callback(gen_ctx, nullptr, nullptr);
    mtmd_input_chunks_free(chunks);

    if (cancelled) {
        // A cancelled prompt is not a failure; the caller asked for this.
        if (on_done) on_done(user);
        return;
    }

    if (eval_result != 0) {
        // An aborted llama_decode() also lands here, so say which it was.
        if (should_cancel_session(session_id)) {
            if (on_done) on_done(user);
        } else if (on_error) {
            on_error("failed to evaluate multimodal prompt", user);
        }
        return;
    }

    vision_stream_from_pos(session_id, new_n_past, on_delta, on_done, on_error, user);
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeIsVisionAvailable(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return llama_vision_available() ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeInitVisionModel(
        JNIEnv *env,
        jobject /*thiz*/,
        jstring jModelPath,
        jstring jProjectionModelPath) {
    const char *model_path = jModelPath ? env->GetStringUTFChars(jModelPath, nullptr) : nullptr;
    const char *projection_model_path = jProjectionModelPath
            ? env->GetStringUTFChars(jProjectionModelPath, nullptr)
            : nullptr;

    const bool success = llama_vision_init(model_path, projection_model_path);

    if (jModelPath && model_path) {
        env->ReleaseStringUTFChars(jModelPath, model_path);
    }
    if (jProjectionModelPath && projection_model_path) {
        env->ReleaseStringUTFChars(jProjectionModelPath, projection_model_path);
    }

    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_llamatik_library_platform_LlamaBridge_nativeGenerateVisionWithMessagesStream(
        JNIEnv *env,
        jobject /*thiz*/,
        jobjectArray jRoles,
        jobjectArray jContents,
        jobjectArray jImagePaths,
        jobject jCallback) {
    if (!jCallback || !jRoles || !jContents || !jImagePaths) {
        LOGE("nativeGenerateVisionWithMessagesStream: null arguments");
        return;
    }

    StreamMethods m{};
    if (!resolve_stream_methods(env, jCallback, m)) {
        LOGE("nativeGenerateVisionWithMessagesStream: failed to resolve callback methods");
        return;
    }

    jsize n_messages = env->GetArrayLength(jRoles);
    jsize n_contents = env->GetArrayLength(jContents);

    if (n_messages != n_contents) {
        env->CallVoidMethod(jCallback, m.onError,
                env->NewStringUTF("roles and contents arrays must have same length"));
        return;
    }

    jsize n_images = env->GetArrayLength(jImagePaths);

    std::vector<std::string> role_strings;
    std::vector<std::string> content_strings;
    std::vector<std::string> image_path_strings;
    std::vector<const char *> roles;
    std::vector<const char *> contents;
    std::vector<const char *> image_paths;
    role_strings.reserve(n_messages);
    content_strings.reserve(n_messages);
    image_path_strings.reserve(n_images);
    roles.reserve(n_messages);
    contents.reserve(n_messages);
    image_paths.reserve(n_images);

    for (jsize i = 0; i < n_messages; i++) {
        jstring jRole = (jstring)env->GetObjectArrayElement(jRoles, i);
        jstring jContent = (jstring)env->GetObjectArrayElement(jContents, i);

        const char *role_cstr = jRole ? env->GetStringUTFChars(jRole, nullptr) : nullptr;
        const char *content_cstr = jContent ? env->GetStringUTFChars(jContent, nullptr) : nullptr;

        role_strings.emplace_back(role_cstr ? role_cstr : "user");
        content_strings.emplace_back(content_cstr ? content_cstr : "");

        if (jRole && role_cstr) env->ReleaseStringUTFChars(jRole, role_cstr);
        if (jContent && content_cstr) env->ReleaseStringUTFChars(jContent, content_cstr);
        if (jRole) env->DeleteLocalRef(jRole);
        if (jContent) env->DeleteLocalRef(jContent);
    }

    for (jsize i = 0; i < n_images; i++) {
        jstring jImagePath = (jstring)env->GetObjectArrayElement(jImagePaths, i);
        const char *image_path_cstr = jImagePath ? env->GetStringUTFChars(jImagePath, nullptr) : nullptr;
        image_path_strings.emplace_back(image_path_cstr ? image_path_cstr : "");
        if (jImagePath && image_path_cstr) env->ReleaseStringUTFChars(jImagePath, image_path_cstr);
        if (jImagePath) env->DeleteLocalRef(jImagePath);
    }

    for (const std::string &role : role_strings) roles.push_back(role.c_str());
    for (const std::string &content : content_strings) contents.push_back(content.c_str());
    for (const std::string &image_path : image_path_strings) image_paths.push_back(image_path.c_str());

    std::pair<JNIEnv *, jobject> callback_context(env, jCallback);
    llama_vision_generate_messages_stream(
            roles.data(),
            contents.data(),
            (int)n_messages,
            image_paths.data(),
            (int)n_images,
            [](const char *utf8, void *user_data) {
                auto *ctx = static_cast<std::pair<JNIEnv *, jobject> *>(user_data);
                StreamMethods methods{};
                if (!resolve_stream_methods(ctx->first, ctx->second, methods)) return;
                jstring delta = ctx->first->NewStringUTF(utf8 ? utf8 : "");
                if (delta) {
                    ctx->first->CallVoidMethod(ctx->second, methods.onDelta, delta);
                    ctx->first->DeleteLocalRef(delta);
                }
            },
            [](void *user_data) {
                auto *ctx = static_cast<std::pair<JNIEnv *, jobject> *>(user_data);
                StreamMethods methods{};
                if (resolve_stream_methods(ctx->first, ctx->second, methods)) {
                    ctx->first->CallVoidMethod(ctx->second, methods.onComplete);
                }
            },
            [](const char *utf8, void *user_data) {
                auto *ctx = static_cast<std::pair<JNIEnv *, jobject> *>(user_data);
                StreamMethods methods{};
                if (!resolve_stream_methods(ctx->first, ctx->second, methods)) return;
                jstring error = ctx->first->NewStringUTF(utf8 ? utf8 : "vision generation failed");
                if (error) {
                    ctx->first->CallVoidMethod(ctx->second, methods.onError, error);
                    ctx->first->DeleteLocalRef(error);
                }
            },
            &callback_context);
}
