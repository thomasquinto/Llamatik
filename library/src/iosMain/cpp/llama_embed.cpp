#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cctype>   // tolower, isalpha
#include <cstdarg>  // va_list, va_start, va_end
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

#ifdef __APPLE__
#include <TargetConditionals.h>
#else
#define TARGET_OS_SIMULATOR 0
#endif

// ===================== Debug logging =====================

static bool g_enable_debug = false;

static void dbg_init() {
    if (g_enable_debug) return;
    const char *e = std::getenv("LLAMATIK_DEBUG");
    g_enable_debug = (e && std::strcmp(e, "0") != 0);
}

static void dbg_printf(const char *fmt, ...) {
    if (!g_enable_debug) return;
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

#define DBG(fmt, ...) \
    do { dbg_printf("[ios] " fmt, ##__VA_ARGS__); } while (0)

// ===================== Error capture =====================

static std::string g_last_error;
static std::mutex g_last_error_mutex;

static void ios_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)user_data;
    if (text == nullptr) return;
    std::string msg(text);
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
    }
    if (msg.empty()) return;

    if (level == GGML_LOG_LEVEL_ERROR) {
        std::lock_guard<std::mutex> lock(g_last_error_mutex);
        g_last_error = msg;
    }
    // Always print to stderr for console/Xcode output
    const char *lvl = (level == GGML_LOG_LEVEL_ERROR) ? "E" :
                       (level == GGML_LOG_LEVEL_WARN)  ? "W" :
                       (level == GGML_LOG_LEVEL_INFO)  ? "I" : "D";
    std::fprintf(stderr, "[llama][%s] %s\n", lvl, msg.c_str());
}

// ===================== Global state =====================

static struct llama_model   *model      = nullptr; // embeddings model
static struct llama_context *ctx        = nullptr;
static int                   embedding_size = 0;

static struct llama_model   *gen_model  = nullptr; // generation model
static struct llama_context *gen_ctx    = nullptr;
static struct mtmd_context  *vision_ctx = nullptr;

static bool g_backend_inited = false;

// Generation session ID - incremented for each new generation
// Used to ensure cancellation only affects the intended generation
static std::atomic<uint64_t> g_generation_session_id{0};

// The session ID that should be cancelled (0 = no cancellation pending)
static std::atomic<uint64_t> g_cancel_session_id{0};

// Flag to track if generation is in progress (for safe shutdown)
static std::atomic<bool> g_generation_in_progress{false};

// Mutex to serialize generation calls (prevents KV cache corruption)
static std::mutex g_generation_mutex;

// Generation parameters (atomic for safe update while app is running)
static std::atomic<float> g_temperature{0.7f};   // align with app default
static std::atomic<int>   g_max_tokens{512};     // align with app default
static std::atomic<float> g_top_p{0.95f};
static std::atomic<int>   g_top_k{40};
static std::atomic<float> g_repeat_penalty{1.10f};

// ===================== Helpers =====================

// Check if the given session should be cancelled
// Returns true if:
// 1. The cancel_session_id matches this session (targeted cancellation), OR
// 2. The cancel_session_id is UINT64_MAX (shutdown/force cancel all)
static inline bool should_cancel_session(uint64_t session_id) {
    uint64_t cancel_id = g_cancel_session_id.load(std::memory_order_relaxed);
    return cancel_id == session_id || cancel_id == UINT64_MAX;
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
    if ((int)tokens.size() <= n_ctx - reserve_tail) return;
    const int keep = n_ctx - reserve_tail;
    std::vector<llama_token> out;
    out.reserve(keep);
    out.insert(out.end(), tokens.end() - keep, tokens.end());
    tokens.swap(out);
}

static llama_model *load_model_with_fallback(const char *path) {
    llama_model_params mp = llama_model_default_params();

#if TARGET_OS_SIMULATOR
    mp.use_mmap     = false;
    mp.use_mlock    = false;
    mp.n_gpu_layers = 0;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE;
#endif

    llama_model *m = llama_model_load_from_file(path, mp);
    if (m) return m;

    mp.use_mmap     = false;
    mp.use_mlock    = false;
    mp.n_gpu_layers = 0;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE;

    return llama_model_load_from_file(path, mp);
}

// ===================== Prompt builders =====================
//
// IMPORTANT: We DO NOT include system instructions verbatim in the prompt text.
// We just structure the task as Context + Question + "Answer:" cue so the model
// finishes the answer without echoing roles.

static std::string build_plain_prompt(const std::string &context_block,
        const std::string &user_msg) {
    std::string p;
    p.reserve(context_block.size() + user_msg.size() + 128);
    if (!context_block.empty()) {
        p += "Context:\n";
        p += context_block;
        p += "\n\n";
    }
    p += "Question:\n";
    p += user_msg;
    p += "\n\nAnswer:\n";
    return p;
}

// No chat template path in this build; keep stub for future wiring.
static bool apply_chat_template_if_available(const char *system_msg,
        const char *user_msg,
        std::string &wrapped) {
    (void)system_msg; (void)user_msg; (void)wrapped;
    return false;
}

// ===================== Text sanitation =====================

static inline std::string trim_ios(std::string s) {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}
static inline std::string to_lower_ios(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c){ return char(std::tolower(c)); });
    return s;
}

static void drop_lines_with_prefix_ci(std::string &s, const char *prefix_ci) {
    std::string out; out.reserve(s.size());
    size_t i = 0, line_start = 0;
    const std::string pfx = to_lower_ios(prefix_ci);
    while (i <= s.size()) {
        if (i == s.size() || s[i] == '\n') {
            std::string line(s.data()+line_start, i - line_start);
            std::string lc = to_lower_ios(line);
            if (!(lc.rfind(pfx, 0) == 0)) {
                out.append(line);
                if (i != s.size()) out.push_back('\n');
            }
            line_start = i + 1;
        }
        ++i;
    }
    s.swap(out);
}

static void drop_lines_containing_ci(std::string &s, const char *needle_ci) {
    std::string out; out.reserve(s.size());
    const std::string ndl = to_lower_ios(needle_ci);
    size_t i = 0, line_start = 0;
    while (i <= s.size()) {
        if (i == s.size() || s[i] == '\n') {
            std::string line(s.data()+line_start, i - line_start);
            std::string lc = to_lower_ios(line);
            if (lc.find(ndl) == std::string::npos) {
                out.append(line);
                if (i != s.size()) out.push_back('\n');
            }
            line_start = i + 1;
        }
        ++i;
    }
    s.swap(out);
}

static std::string sanitize_generation_ios(std::string s) {
    if (s.empty()) return s;

    // 1) cut at common EOT / next-turn markers
    for (const char* stop : { "<end_of_turn>", "<|eot_id|>", "</s>", "<start_of_turn>" }) {
        size_t p = s.find(stop);
        if (p != std::string::npos) { s = s.substr(0, p); }
    }

    // 2) remove leaked role headers and common labels
    for (const char* pfx : { "assistant:", "user:", "system:", "answer:" }) {
        drop_lines_with_prefix_ci(s, pfx);
    }

    // 3) strip any accidental copies of your system guidance
    for (const char* sub : {
            "you are a helpful technical assistant",
            "answer in plain text",
            "do not echo the question",
            "never write role labels"
    }) {
        drop_lines_containing_ci(s, sub);
    }

    // 4) if we still see a lone "Answer:" cue, remove that cue only
    {
        std::string low = to_lower_ios(s);
        size_t p = low.find("answer:");
        if (p != std::string::npos) {
            // delete "Answer:" and any immediate single space
            size_t end = p + 7;
            if (end < s.size() && s[end] == ' ') ++end;
            s.erase(p, end - p);
        }
    }

    // 5) trim
    s = trim_ios(s);
    return s;
}

// === Streaming: robust gate so we don't show partial headers or "Answer:" ===

// Return the index where real content starts, or npos if we should wait for more chars.
static size_t find_stream_start(const std::string &s) {
    size_t i = 0;

    auto is_space = [](char c){ return c==' '||c=='\t'||c=='\r'||c=='\n'; };
    auto starts_ci = [&](size_t pos, const char* w)->bool{
        size_t n = std::strlen(w);
        if (pos + n > s.size()) return false;
        for (size_t k = 0; k < n; ++k) {
            char a = std::tolower((unsigned char)s[pos+k]);
            char b = std::tolower((unsigned char)w[k]);
            if (a != b) return false;
        }
        return true;
    };
    auto is_prefix_ci = [&](size_t pos, const char* w)->bool{
        size_t n = std::strlen(w);
        size_t len = std::min(n, s.size() - pos);
        for (size_t k = 0; k < len; ++k) {
            char a = std::tolower((unsigned char)s[pos+k]);
            char b = std::tolower((unsigned char)w[k]);
            if (a != b) return false;
        }
        return true; // s[pos..] matches the prefix of w
    };

    // skip leading whitespace
    while (i < s.size() && is_space(s[i])) ++i;

    while (i < s.size()) {
        // Incomplete or complete tag line: wait until '>' then skip it (and trailing spaces/newline)
        if (s[i] == '<') {
            size_t gt = s.find('>', i + 1);
            size_t nl = s.find('\n', i);
            if (gt == std::string::npos || (nl != std::string::npos && nl < gt)) {
                return std::string::npos; // incomplete tag line
            }
            i = gt + 1;
            while (i < s.size() && is_space(s[i])) ++i;
            continue;
        }

        // Handle role labels (assistant|user|system|answer), even if partial
        if (is_prefix_ci(i, "assistant") || is_prefix_ci(i, "user") ||
                is_prefix_ci(i, "system") || is_prefix_ci(i, "answer")) {
            // If we don't yet have the full word, wait.
            if (!(starts_ci(i, "assistant") || starts_ci(i, "user") ||
                    starts_ci(i, "system") || starts_ci(i, "answer"))) {
                return std::string::npos; // partial like "Assis" or "Ans"
            }
            // We have the full word; if colon not here yet, wait one more char.
            size_t j = i;
            while (j < s.size() && std::isalpha((unsigned char)s[j])) ++j;
            if (j >= s.size()) return std::string::npos; // need more to see ':' or content

            if (s[j] == ':') {
                // Skip "Label:" + spaces and an optional newline, then continue scanning
                ++j;
                while (j < s.size() && (s[j] == ' ' || s[j] == '\t')) ++j;
                if (j < s.size() && s[j] == '\n') {
                    ++j;
                    while (j < s.size() && is_space(s[j])) ++j;
                }
                i = j;
                continue; // drop the label and keep looking
            }
            // Full word but no colon: treat as normal content (rare)
            return i;
        }

        // Otherwise, content starts here.
        return i;
    }

    return std::string::npos;
}

// ===================== Embeddings =====================

extern "C" {

bool llama_embed_init(const char *model_path) {
    dbg_init();
    if (!g_backend_inited) {
        llama_backend_init();
        g_backend_inited = true;
    }

    model = load_model_with_fallback(model_path);
    if (!model) return false;

    llama_context_params cp = llama_context_default_params();
    cp.embeddings = true;
    cp.n_ctx      = 2048;

    ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        model = nullptr;
        return false;
    }

    embedding_size = llama_model_n_embd(model);
    DBG("embed: dim=%d", embedding_size);
    return true;
}

void llama_generate_cancel(void) {
    // Cancel the CURRENT session only
    // This prevents race conditions where a late cancellation affects a new generation
    uint64_t current_session = g_generation_session_id.load(std::memory_order_relaxed);
    DBG("llama_generate_cancel: requesting cancel for session %llu", (unsigned long long)current_session);
    g_cancel_session_id.store(current_session, std::memory_order_relaxed);
}

float *llama_embed(const char *input) {
    if (!ctx || !model || !input) return nullptr;

    std::vector<llama_token> tokens(1024);
    int n_tokens = tokenize_with_retry(
            llama_model_get_vocab(model),
            input,
            tokens,
            /*add_bos*/ true,
            /*parse_special*/ false);

    if (n_tokens <= 0 || n_tokens > llama_n_ctx(ctx)) {
        DBG("embed: tokenize fail/too long n=%d ctx=%u", n_tokens, (unsigned)llama_n_ctx(ctx));
        return nullptr;
    }
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = false;
    }

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        DBG("embed: decode failed");
        return nullptr;
    }

    const float *emb = llama_get_embeddings_seq(ctx, 0);
    if (!emb) {
        llama_batch_free(batch);
        DBG("embed: embeddings null");
        return nullptr;
    }

    const int dim = llama_model_n_embd(model);
    float *out = (float *) std::malloc(sizeof(float) * (size_t)dim);
    if (!out) {
        llama_batch_free(batch);
        return nullptr;
    }
    std::memcpy(out, emb, sizeof(float) * (size_t)dim);
    llama_batch_free(batch);
    return out;
}

int   llama_embedding_size()          { return llama_model_n_embd(model); }
void  llama_free_embedding(float *p)  { if (p) std::free(p); }

void llama_embed_free() {
    if (ctx)   llama_free(ctx);
    if (model) llama_model_free(model);
    ctx = nullptr; model = nullptr;

    if (!gen_ctx && !gen_model && !vision_ctx && g_backend_inited) {
        llama_backend_free();
        g_backend_inited = false;
    }
}

// ===================== Text Generation =====================

bool llama_generate_init(const char *model_path) {
    dbg_init();
    if (!g_backend_inited) {
        llama_log_set(ios_log_callback, nullptr);
        llama_backend_init();
        g_backend_inited = true;
    }

    // Clear last error before loading
    {
        std::lock_guard<std::mutex> lock(g_last_error_mutex);
        g_last_error.clear();
    }

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

    gen_model = load_model_with_fallback(model_path);
    if (!gen_model) return false;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = false;
    ctx_params.n_ctx      = 8192;   // larger context

    gen_ctx = llama_init_from_model(gen_model, ctx_params);
    if (!gen_ctx) {
        llama_model_free(gen_model);
        gen_model = nullptr;
        return false;
    }
    DBG("generate: n_ctx = %u", (unsigned)llama_n_ctx(gen_ctx));
    return true;
}

const char *llama_generate_last_error(void) {
    std::lock_guard<std::mutex> lock(g_last_error_mutex);
    if (g_last_error.empty()) return nullptr;
    // Return pointer to static string - valid until next error
    return g_last_error.c_str();
}

char *llama_generate(const char *prompt) {
    if (!gen_ctx || !gen_model || !prompt) return nullptr;

    // Assign a new session ID for this generation
    uint64_t session_id = ++g_generation_session_id;

    // Snapshot params at start (atomic -> local)
    const float temperature    = g_temperature.load(std::memory_order_relaxed);
    const int   max_tokens     = g_max_tokens.load(std::memory_order_relaxed);
    const float top_p          = g_top_p.load(std::memory_order_relaxed);
    const int   top_k          = g_top_k.load(std::memory_order_relaxed);
    const float repeat_penalty = g_repeat_penalty.load(std::memory_order_relaxed);

    // Clear any stale cancellation that was for a previous session
    // Clear if: cancel was for an older session, OR it's UINT64_MAX (shutdown signal)
    // UINT64_MAX must be cleared because it cancels ALL sessions, including new ones
    uint64_t cancel_id = g_cancel_session_id.load(std::memory_order_relaxed);
    if (cancel_id != 0 && (cancel_id < session_id || cancel_id == UINT64_MAX)) {
        g_cancel_session_id.store(0, std::memory_order_relaxed);
    }

    llama_memory_clear(llama_get_memory(gen_ctx), false);

    // Use prompt as-is (caller is responsible for formatting)
    const llama_vocab *v = llama_model_get_vocab(gen_model);
    std::vector<llama_token> tokens(2048);
    int n_tokens = tokenize_with_retry(v, prompt, tokens, /*add_bos*/ true, /*parse_special*/ true);
    if (n_tokens <= 0) return nullptr;
    tokens.resize(n_tokens);

    const unsigned int n_ctx = llama_n_ctx(gen_ctx);
    if (n_tokens > (int) n_ctx - 8) {
        truncate_to_ctx(tokens, (int) n_ctx, 8);
        DBG("generate: prompt truncated");
    }

    llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
    batch.n_tokens = (int)tokens.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == batch.n_tokens - 1);
    }

    if (llama_decode(gen_ctx, batch) != 0) {
        llama_batch_free(batch);
        DBG("generate: decode prompt failed");
        return nullptr;
    }

    // Sampler
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) {
        llama_batch_free(batch);
        return nullptr;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // 3) Decode loop
    std::vector<llama_token> out;
    int cur_pos = batch.n_tokens;
    const int safety = 16;
    int remaining_ctx = (int)n_ctx - cur_pos - safety;
    if (remaining_ctx < 0) remaining_ctx = 0;

    // IMPORTANT: limit to what's left in ctx AND the user-configured max_tokens
    int max_new_tokens = std::min(remaining_ctx, max_tokens);

    for (int i = 0; i < max_new_tokens; ++i) {
        // Check if THIS session should be cancelled (not a different one)
        if (should_cancel_session(session_id)) {
            DBG("generate: cancelled session %llu at token %d", (unsigned long long)session_id, i);
            break;
        }

        llama_token tok = llama_sampler_sample(sampler, gen_ctx, -1);
        if (tok < 0) break;
        if (llama_vocab_is_eog(v, tok)) {
            DBG("generate: hit EOS");
            break;
        }

        // Early stop on common “end” pieces
        char piece[64];
        int nn = llama_token_to_piece(v, tok, piece, (int)sizeof(piece), 0, /*special*/ true);
        if (nn > 0) {
            if (nn >= (int)sizeof(piece)) piece[sizeof(piece)-1] = '\0';
            else piece[nn] = '\0';
            if (std::strcmp(piece, "<|eot_id|>") == 0 ||
                    std::strcmp(piece, "<end_of_turn>") == 0 ||
                    std::strcmp(piece, "</s>") == 0 ||
                    std::strcmp(piece, "<start_of_turn>") == 0) {
                DBG("generate: hit EOT piece: %s", piece);
                break;
            }
        }

        llama_sampler_accept(sampler, tok);
        out.push_back(tok);

        if (cur_pos >= (int)n_ctx) {
            DBG("generate: context full at %d positions", cur_pos);
            break;
        }

        llama_batch step = llama_batch_init(1, 0, 1);
        step.n_tokens      = 1;
        step.token[0]      = tok;
        step.pos[0]        = cur_pos;
        step.n_seq_id[0]   = 1;
        step.seq_id[0][0]  = 0;
        step.logits[0]     = true;

        if (llama_decode(gen_ctx, step) != 0) {
            DBG("generate: decode step failed at pos=%d", cur_pos);
            llama_batch_free(step);
            break;
        }
        cur_pos++;
        llama_batch_free(step);
    }

    llama_batch_free(batch);
    llama_sampler_free(sampler);

    // 4) Detokenize
    std::string text;
    char buf[8192];
    for (llama_token t : out) {
        int n = llama_token_to_piece(v, t, buf, (int)sizeof(buf), 0, /*special*/ false);
        if (n > 0) {
            if (n >= (int)sizeof(buf)) buf[sizeof(buf)-1] = '\0';
            text.append(buf, n);
        }
    }

    // Strong sanitize (remove any leaked labels / system echoes)
    text = sanitize_generation_ios(std::move(text));

    char *result = (char *) std::malloc(text.size() + 1);
    if (!result) return nullptr;
    std::memcpy(result, text.c_str(), text.size() + 1);
    return result;
}

// Clean prompt helper used by chat wrapper
static std::string build_clean_prompt(const char *system_prompt,
        const char *context_block,
        const char *user_prompt) {
    (void)system_prompt; // not injected into text; we keep only a simple structure
    std::string ctxb = context_block ? context_block : "";
    std::string usr = user_prompt   ? user_prompt   : "";
    return build_plain_prompt(ctxb, usr);
}

char *llama_generate_chat(const char *system_prompt,
        const char *context_block,
        const char *user_prompt) {
    std::string prompt2 = build_clean_prompt(system_prompt, context_block, user_prompt);
    char *raw = llama_generate(prompt2.c_str());
    return raw; // already sanitized inside llama_generate
}

// ===================== Streaming APIs (iOS) =====================

typedef void (*llm_on_delta)(const char *utf8, void *user);
typedef void (*llm_on_done)(void *user);
typedef void (*llm_on_error)(const char *utf8, void *user);

void llama_generate_stream(const char *prompt,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user) {
    // Serialize generation calls to prevent KV cache corruption
    std::lock_guard<std::mutex> lock(g_generation_mutex);

    // Assign a new session ID for this generation
    // This must happen AFTER acquiring the mutex to ensure uniqueness
    uint64_t session_id = ++g_generation_session_id;
    DBG("llama_generate_stream: session_id=%llu, cancel_session=%llu",
        (unsigned long long)session_id,
        (unsigned long long)g_cancel_session_id.load());

    if (!gen_ctx || !gen_model || !prompt) { if (on_error) on_error("generator not ready", user); return; }

    // Mark generation as in progress (for safe shutdown)
    g_generation_in_progress.store(true, std::memory_order_release);

    // Clear any stale cancellation that was for a previous session
    // Clear if: cancel was for an older session, OR it's UINT64_MAX (shutdown signal)
    // UINT64_MAX must be cleared because it cancels ALL sessions, including new ones
    uint64_t cancel_id = g_cancel_session_id.load(std::memory_order_relaxed);
    if (cancel_id != 0 && (cancel_id < session_id || cancel_id == UINT64_MAX)) {
        g_cancel_session_id.store(0, std::memory_order_relaxed);
        DBG("llama_generate_stream: cleared stale cancel (was %llu)", (unsigned long long)cancel_id);
    }

    // Snapshot params at start (atomic -> local)
    const float temperature    = g_temperature.load(std::memory_order_relaxed);
    const int   max_tokens     = g_max_tokens.load(std::memory_order_relaxed);
    const float top_p          = g_top_p.load(std::memory_order_relaxed);
    const int   top_k          = g_top_k.load(std::memory_order_relaxed);
    const float repeat_penalty = g_repeat_penalty.load(std::memory_order_relaxed);

    llama_memory_clear(llama_get_memory(gen_ctx), false);

    // Use prompt as-is (caller is responsible for formatting)
    std::vector<llama_token> tokens(2048);
    int n_tokens = tokenize_with_retry(llama_model_get_vocab(gen_model),
            prompt,
            tokens, /*add_bos*/ true, /*parse_special*/ true);
    if (n_tokens <= 0) {
        g_generation_in_progress.store(false, std::memory_order_release);
        if (on_error) on_error("tokenize failed", user);
        return;
    }
    tokens.resize(n_tokens);

    const unsigned int n_ctx = llama_n_ctx(gen_ctx);
    if (n_tokens > (int)n_ctx - 8) {
        truncate_to_ctx(tokens, (int)n_ctx, 8);
    }

    llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
    batch.n_tokens = (int)tokens.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == batch.n_tokens - 1);
    }

    if (llama_decode(gen_ctx, batch) != 0) {
        llama_batch_free(batch);
        g_generation_in_progress.store(false, std::memory_order_release);
        if (on_error) on_error("decode failed", user);
        return;
    }

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) {
        llama_batch_free(batch);
        g_generation_in_progress.store(false, std::memory_order_release);
        if (on_error) on_error("sampler init failed", user);
        return;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_vocab *v = llama_model_get_vocab(gen_model);

    int cur_pos = batch.n_tokens;
    const int safety = 16;
    int remaining_ctx = (int)n_ctx - cur_pos - safety;
    if (remaining_ctx < 0) remaining_ctx = 0;

    // IMPORTANT: limit to what's left in ctx AND the user-configured max_tokens
    int max_new_tokens = std::min(remaining_ctx, max_tokens);

    std::string assembled;            // raw accumulation from tokens (no specials)
    size_t start_idx = std::string::npos; // where real content begins
    size_t sent_from_start = 0;           // how many chars emitted from [start_idx..)

    assembled.reserve(4096);

    for (int i = 0; i < max_new_tokens; ++i) {
        // Check if THIS session should be cancelled (not a different one)
        if (should_cancel_session(session_id)) {
            DBG("stream: cancelled session %llu at token %d", (unsigned long long)session_id, i);
            break;
        }

        llama_token tok = llama_sampler_sample(sampler, gen_ctx, -1);
        if (tok < 0) break;
        if (llama_vocab_is_eog(v, tok)) break;

        // Early stop on common “end” pieces (including turn tags)
        char spiece[64];
        int nn = llama_token_to_piece(v, tok, spiece, (int)sizeof(spiece), 0, /*special*/ true);
        if (nn > 0) {
            if (nn >= (int)sizeof(spiece)) spiece[sizeof(spiece)-1] = '\0';
            else spiece[nn] = '\0';
            if (std::strcmp(spiece, "<|eot_id|>") == 0 ||
                    std::strcmp(spiece, "<end_of_turn>") == 0 ||
                    std::strcmp(spiece, "</s>") == 0 ||
                    std::strcmp(spiece, "<start_of_turn>") == 0) {
                break;
            }
        }

        llama_sampler_accept(sampler, tok);

        // Append normal text piece (no specials) to assembled buffer
        char piece[256];
        int nout = llama_token_to_piece(v, tok, piece, (int)sizeof(piece), 0, /*special*/ false);
        if (nout > 0) {
            if (nout >= (int)sizeof(piece)) piece[sizeof(piece)-1] = '\0';
            assembled.append(piece, nout);

            // Try to find a safe start (skip role labels / "Answer:" / tags)
            if (start_idx == std::string::npos) {
                start_idx = find_stream_start(assembled);
            }

            // If we have a safe start, emit only the *new* tail from that point.
            if (start_idx != std::string::npos && assembled.size() > start_idx + sent_from_start) {
                const std::string_view delta(assembled.data() + start_idx + sent_from_start,
                        assembled.size() - (start_idx + sent_from_start));
                if (on_delta && !delta.empty()) on_delta(std::string(delta).c_str(), user);
                sent_from_start += delta.size();
            }
        }

        if (cur_pos >= (int)n_ctx) break;

        llama_batch step = llama_batch_init(1, 0, 1);
        step.n_tokens      = 1;
        step.token[0]      = tok;
        step.pos[0]        = cur_pos;
        step.n_seq_id[0]   = 1;
        step.seq_id[0][0]  = 0;
        step.logits[0]     = true;
        if (llama_decode(gen_ctx, step) != 0) {
            llama_batch_free(step);
            break;
        }
        cur_pos++;
        llama_batch_free(step);
    }

    llama_batch_free(batch);
    llama_sampler_free(sampler);

    // Mark generation as complete (for safe shutdown)
    g_generation_in_progress.store(false, std::memory_order_release);

    if (on_done) on_done(user);
}

void llama_generate_chat_stream(const char *system_prompt,
        const char *context_block,
        const char *user_prompt,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user) {
    std::string prompt2 = build_clean_prompt(system_prompt ? system_prompt : "",
            context_block ? context_block : "",
            user_prompt ? user_prompt : "");
    llama_generate_stream(prompt2.c_str(), on_delta, on_done, on_error, user);
}

void llama_generate_set_params(float temperature,
        int max_tokens,
        float top_p,
        int top_k,
        float repeat_penalty) {
    g_temperature.store(temperature, std::memory_order_relaxed);
    g_max_tokens.store(max_tokens, std::memory_order_relaxed);
    g_top_p.store(top_p, std::memory_order_relaxed);
    g_top_k.store(top_k, std::memory_order_relaxed);
    g_repeat_penalty.store(repeat_penalty, std::memory_order_relaxed);
}

void llama_generate_free() {
    DBG("llama_generate_free: starting, generation_in_progress=%d", g_generation_in_progress.load());

    // Signal cancellation for ALL sessions (force shutdown)
    // UINT64_MAX is a special value that cancels any session
    g_cancel_session_id.store(UINT64_MAX, std::memory_order_release);

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
        DBG("llama_generate_free: WARNING - generation did not stop within timeout");
    } else {
        DBG("llama_generate_free: generation stopped after %d ms", wait_count * sleep_ms);
    }

    if (gen_ctx)   llama_free(gen_ctx);
    if (gen_model) llama_model_free(gen_model);
    if (vision_ctx) mtmd_free(vision_ctx);
    gen_ctx   = nullptr;
    gen_model = nullptr;
    vision_ctx = nullptr;

    if (!ctx && !model && g_backend_inited) {
        llama_backend_free();
        g_backend_inited = false;
    }

    DBG("llama_generate_free: complete");
}

// ===================== Chat Template Streaming (Message Array API) =====================

/**
 * Build a prompt using llama.cpp's chat template system.
 * Uses the model's embedded template if available, otherwise falls back to ChatML.
 *
 * @param roles Array of role strings ("system", "user", "assistant")
 * @param contents Array of content strings
 * @param n_messages Number of messages
 * @return Formatted prompt string ready for tokenization
 */
static std::string build_chat_template_prompt_ios(const char **roles,
        const char **contents,
        int n_messages) {
    if (!gen_model || n_messages <= 0) {
        DBG("build_chat_template_prompt_ios: model not loaded or no messages");
        return "";
    }

    // Build llama_chat_message array
    std::vector<llama_chat_message> messages;
    messages.reserve(n_messages);
    for (int i = 0; i < n_messages; i++) {
        messages.push_back({roles[i], contents[i]});
    }

    // Get the model's embedded chat template (may be nullptr)
    const char *model_template = llama_model_chat_template(gen_model, nullptr);

    if (model_template) {
        DBG("Using model's embedded chat template");
    } else {
        DBG("Model has no embedded template, using ChatML fallback");
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
            DBG("llama_chat_apply_template failed for %s with error %d", label, apply_result);
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
                DBG("llama_chat_apply_template retry failed for %s with error %d", label, apply_result);
            }
        }

        return apply_result;
    };

    int32_t result = apply_template(model_template, model_template ? "embedded template" : "ChatML fallback");

    if (result < 0 && model_template && is_gemma_family) {
        DBG("Embedded chat template is not supported for %s; retrying with Gemma template", architecture.c_str());
        result = apply_template("gemma", "Gemma fallback");
    }

    if (result < 0 && model_template && !is_gemma_family) {
        DBG("Embedded chat template is not supported for %s; retrying with ChatML fallback", architecture.c_str());
        result = apply_template(nullptr, "ChatML fallback");
    }

    if (result < 0) {
        DBG("Unable to format chat prompt with embedded/default/fallback templates");
        return "";
    }

    std::string prompt(buf.data(), result);
    DBG("Chat template prompt (%d chars): %.100s...", result, prompt.c_str());
    return prompt;
}

/**
 * Stream generation using chat template with message array.
 *
 * @param roles Array of role strings ("system", "user", "assistant")
 * @param contents Array of content strings
 * @param n_messages Number of messages
 * @param on_delta Callback for each token
 * @param on_done Callback when complete
 * @param on_error Callback on error
 * @param user User data pointer
 */
void llama_generate_messages_stream(const char **roles,
        const char **contents,
        int n_messages,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user) {
    if (!roles || !contents || n_messages <= 0) {
        if (on_error) on_error("invalid arguments: roles, contents, or n_messages", user);
        return;
    }

    std::string prompt = build_chat_template_prompt_ios(roles, contents, n_messages);

    if (prompt.empty()) {
        if (on_error) on_error("failed to build chat template prompt", user);
        return;
    }

    // Use existing stream function with the templated prompt
    llama_generate_stream(prompt.c_str(), on_delta, on_done, on_error, user);
}

bool llama_vision_available(void) {
    return true;
}

bool llama_vision_init(const char *model_path, const char *projection_model_path) {
    dbg_init();
    if (!model_path || !projection_model_path) {
        DBG("llama_vision_init: missing model or projection path");
        return false;
    }

    if (!g_backend_inited) {
        llama_log_set(ios_log_callback, nullptr);
        llama_backend_init();
        g_backend_inited = true;
    }

    if (!gen_model || !gen_ctx) {
        gen_model = load_model_with_fallback(model_path);
        if (!gen_model) {
            DBG("llama_vision_init: failed to load base model");
            return false;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.embeddings = false;
        ctx_params.n_ctx = 8192;
        gen_ctx = llama_init_from_model(gen_model, ctx_params);
        if (!gen_ctx) {
            DBG("llama_vision_init: failed to create base model context");
            llama_model_free(gen_model);
            gen_model = nullptr;
            return false;
        }
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
        DBG("llama_vision_init: failed to load projector");
        return false;
    }

    if (!mtmd_support_vision(vision_ctx)) {
        DBG("llama_vision_init: projector does not report vision support");
        mtmd_free(vision_ctx);
        vision_ctx = nullptr;
        return false;
    }

    return true;
}

static std::string build_vision_prompt_ios(
        const char **roles,
        const char **contents,
        int n_messages,
        int n_images) {
    std::vector<std::string> role_strings;
    std::vector<std::string> content_strings;
    std::vector<const char *> role_ptrs;
    std::vector<const char *> content_ptrs;
    role_strings.reserve(n_messages);
    content_strings.reserve(n_messages);
    role_ptrs.reserve(n_messages);
    content_ptrs.reserve(n_messages);

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
        role_ptrs.push_back(role_strings.back().c_str());
        content_ptrs.push_back(content_strings.back().c_str());
    }

    return build_chat_template_prompt_ios(role_ptrs.data(), content_ptrs.data(), n_messages);
}

static void llama_vision_stream_from_pos(
        uint64_t session_id,
        llama_pos start_pos,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user) {
    const float temperature = g_temperature.load(std::memory_order_relaxed);
    const int max_tokens = g_max_tokens.load(std::memory_order_relaxed);
    const float top_p = g_top_p.load(std::memory_order_relaxed);
    const int top_k = g_top_k.load(std::memory_order_relaxed);
    const float repeat_penalty = g_repeat_penalty.load(std::memory_order_relaxed);

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) {
        if (on_error) on_error("sampler init failed", user);
        return;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(128, repeat_penalty, 0.0f, 0.10f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_vocab *v = llama_model_get_vocab(gen_model);
    const int n_ctx = (int)llama_n_ctx(gen_ctx);
    int cur_pos = (int)start_pos;
    const int max_new_tokens = std::min(max_tokens, std::max(0, n_ctx - cur_pos - 16));
    char piece[256];
    char spiece[64];

    for (int i = 0; i < max_new_tokens; ++i) {
        if (should_cancel_session(session_id)) break;

        llama_token tok = llama_sampler_sample(sampler, gen_ctx, -1);
        if (tok < 0 || llama_vocab_is_eog(v, tok)) break;

        int sn = llama_token_to_piece(v, tok, spiece, (int)sizeof(spiece), 0, true);
        if (sn > 0) {
            spiece[std::min(sn, (int)sizeof(spiece) - 1)] = '\0';
            if (std::strcmp(spiece, "<|eot_id|>") == 0 ||
                    std::strcmp(spiece, "<end_of_turn>") == 0 ||
                    std::strcmp(spiece, "</s>") == 0 ||
                    std::strcmp(spiece, "<start_of_turn>") == 0) {
                break;
            }
        }

        llama_sampler_accept(sampler, tok);

        int nout = llama_token_to_piece(v, tok, piece, (int)sizeof(piece), 0, false);
        if (nout > 0 && on_delta) {
            piece[std::min(nout, (int)sizeof(piece) - 1)] = '\0';
            on_delta(piece, user);
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
            if (on_error) on_error("decode step failed", user);
            return;
        }
        llama_batch_free(step);
    }

    llama_sampler_free(sampler);
    if (on_done) on_done(user);
}

void llama_vision_generate_messages_stream(const char **roles,
        const char **contents,
        int n_messages,
        const char **image_paths,
        int n_images,
        llm_on_delta on_delta,
        llm_on_done on_done,
        llm_on_error on_error,
        void *user) {
    std::lock_guard<std::mutex> lock(g_generation_mutex);
    const uint64_t session_id = ++g_generation_session_id;

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

    uint64_t cancel_id = g_cancel_session_id.load(std::memory_order_relaxed);
    if (cancel_id != 0 && (cancel_id < session_id || cancel_id == UINT64_MAX)) {
        g_cancel_session_id.store(0, std::memory_order_relaxed);
    }

    g_generation_in_progress.store(true, std::memory_order_release);
    llama_memory_clear(llama_get_memory(gen_ctx), true);

    std::vector<mtmd_bitmap *> bitmaps;
    std::vector<const mtmd_bitmap *> bitmap_refs;
    bitmaps.reserve(n_images);
    bitmap_refs.reserve(n_images);
    for (int i = 0; i < n_images; ++i) {
        auto bitmap = mtmd_helper_bitmap_init_from_file(vision_ctx, image_paths[i], false);
        if (!bitmap.bitmap) {
            for (mtmd_bitmap *loaded : bitmaps) mtmd_bitmap_free(loaded);
            g_generation_in_progress.store(false, std::memory_order_release);
            if (on_error) on_error("failed to load image", user);
            return;
        }
        bitmaps.push_back(bitmap.bitmap);
        bitmap_refs.push_back(bitmap.bitmap);
        if (bitmap.video_ctx) {
            mtmd_helper_video_free(bitmap.video_ctx);
        }
    }

    std::string prompt = build_vision_prompt_ios(roles, contents, n_messages, n_images);
    if (prompt.empty()) {
        for (mtmd_bitmap *bitmap : bitmaps) mtmd_bitmap_free(bitmap);
        g_generation_in_progress.store(false, std::memory_order_release);
        if (on_error) on_error("failed to build chat template prompt", user);
        return;
    }

    mtmd_input_text text{};
    text.text = prompt.c_str();
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
        g_generation_in_progress.store(false, std::memory_order_release);
        if (on_error) on_error("failed to tokenize multimodal prompt", user);
        return;
    }

    llama_pos new_n_past = 0;
    int32_t eval_result = mtmd_helper_eval_chunks(
            vision_ctx,
            gen_ctx,
            chunks,
            0,
            0,
            512,
            true,
            &new_n_past);
    mtmd_input_chunks_free(chunks);

    if (eval_result != 0) {
        g_generation_in_progress.store(false, std::memory_order_release);
        if (on_error) on_error("failed to evaluate multimodal prompt", user);
        return;
    }

    llama_vision_stream_from_pos(session_id, new_n_past, on_delta, on_done, on_error, user);
    g_generation_in_progress.store(false, std::memory_order_release);
}

} // extern "C"
