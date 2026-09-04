#include "engine_state.h"

#include <atomic>
#include <mutex>

namespace llamatik {
namespace {

// Incremented per generation so a cancellation can name the one it means.
std::atomic<uint64_t> g_generation_session_id{0};

// The session to cancel. 0 means none; UINT64_MAX means all of them.
std::atomic<uint64_t> g_cancel_session_id{0};

std::atomic<bool> g_generation_in_progress{false};

// Serialises generation against teardown, and against itself: concurrent generations on
// one context corrupt the KV cache.
std::mutex g_generation_mutex;

// Guards g_resident_model_path. Separate from the generation mutex, which is held for
// entire generations and would make reading the path block for half a minute.
std::mutex g_resident_path_mutex;
std::string g_resident_model_path;

constexpr uint64_t kCancelNone = 0;
constexpr uint64_t kCancelAll = UINT64_MAX;

} // namespace

uint64_t begin_session() {
    const uint64_t session_id = ++g_generation_session_id;

    // Drop a cancellation aimed at an older session, or a force-cancel-everything left by
    // a shutdown. Either would otherwise apply to this session, which never asked for it.
    const uint64_t pending = g_cancel_session_id.load(std::memory_order_relaxed);
    if (pending != kCancelNone && (pending < session_id || pending == kCancelAll)) {
        g_cancel_session_id.store(kCancelNone, std::memory_order_relaxed);
    }
    return session_id;
}

bool should_cancel_session(uint64_t session_id) {
    const uint64_t pending = g_cancel_session_id.load(std::memory_order_relaxed);
    return pending == session_id || pending == kCancelAll;
}

void cancel_session(uint64_t session_id) {
    g_cancel_session_id.store(session_id, std::memory_order_release);
}

void cancel_all_sessions() {
    g_cancel_session_id.store(kCancelAll, std::memory_order_release);
}

GenerationLock::GenerationLock() {
    g_generation_mutex.lock();
    g_generation_in_progress.store(true, std::memory_order_release);
}

GenerationLock::~GenerationLock() {
    g_generation_in_progress.store(false, std::memory_order_release);
    g_generation_mutex.unlock();
}

void stop_generation_and_wait() {
    cancel_all_sessions();
    // Blocks until generation releases the lock. That is the whole point: a timeout here
    // would expire during a long image encode and free the model out from under it.
    std::lock_guard<std::mutex> lock(g_generation_mutex);
}

bool generation_in_progress() {
    return g_generation_in_progress.load(std::memory_order_acquire);
}

bool can_reuse_resident_model(bool a_model_is_loaded, const std::string &model_path) {
    if (!a_model_is_loaded) return false;
    std::lock_guard<std::mutex> lock(g_resident_path_mutex);
    return !g_resident_model_path.empty() && g_resident_model_path == model_path;
}

void set_resident_model_path(const std::string &model_path) {
    std::lock_guard<std::mutex> lock(g_resident_path_mutex);
    g_resident_model_path = model_path;
}

void clear_resident_model_path() {
    std::lock_guard<std::mutex> lock(g_resident_path_mutex);
    g_resident_model_path.clear();
}

const std::string &resident_model_path() {
    // Safe to return by reference: the string outlives every caller, and callers compare
    // or copy it rather than hold it across a load.
    std::lock_guard<std::mutex> lock(g_resident_path_mutex);
    return g_resident_model_path;
}

} // namespace llamatik
