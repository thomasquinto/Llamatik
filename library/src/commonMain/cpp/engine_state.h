#pragma once

// Generation lifecycle state, shared by the Android and iOS bridges.
//
// Android drives llama.cpp through JNI (llama_jni.cpp) and iOS through a C API
// (iosMain/cpp/llama_embed.cpp). Those entry points differ; the rules about when a model
// may be freed, when a generation may be cancelled, and whether the resident model is the
// one being asked for do not. Keeping two copies of those rules produced the same bug
// twice, three separate times, and each was fixed on one platform only.
//
// Deliberately depends on nothing but the standard library — no llama.cpp types, no JNI,
// no Foundation — so it compiles into both bridges and can be tested on the host without
// a model, a device, or a framework.

#include <cstdint>
#include <string>

namespace llamatik {

// ---------------------------------------------------------------------------------
// Sessions and cancellation
// ---------------------------------------------------------------------------------

/// Begins a generation and returns its session ID.
///
/// Also clears a cancellation left over from an earlier session. Without that, a
/// shutdown's force-cancel stays pending and immediately kills the next generation —
/// which looks like generation silently doing nothing.
uint64_t begin_session();

/// Whether `session_id` should stop: either it was cancelled by ID, or everything was.
bool should_cancel_session(uint64_t session_id);

/// Requests that `session_id` stop. Advisory: llama.cpp checks it between tokens, and
/// not at all inside an image encode.
void cancel_session(uint64_t session_id);

/// Requests that every session stop, including any that starts before the flag is cleared.
void cancel_all_sessions();

/// The most recently started session, or 0 if none has started.
///
/// For cancelling "whatever is running now". Inherently a snapshot: the session may end
/// between reading this and acting on it, which is why cancellation is by ID — a late
/// cancel names a session that is already gone and is ignored.
uint64_t current_session();

/// The session cancellation is pending for: 0 for none, UINT64_MAX for all. Diagnostics only.
uint64_t pending_cancel_session();

// ---------------------------------------------------------------------------------
// Serialising generation against teardown
// ---------------------------------------------------------------------------------

/// Held for the whole of a generation. Freeing the model without it is a use-after-free.
class GenerationLock {
public:
    GenerationLock();
    ~GenerationLock();
    GenerationLock(const GenerationLock &) = delete;
    GenerationLock &operator=(const GenerationLock &) = delete;
};

/// Cancels generation and then holds the generation lock for its own lifetime.
///
/// Freeing a model requires holding this across the free, not merely waiting first:
/// releasing the lock before tearing down reopens the window where a generation can start
/// on a context that is about to disappear. Unlike GenerationLock this does not mark
/// generation as in progress, because teardown is not generation.
class TeardownLock {
public:
    TeardownLock();
    ~TeardownLock();
    TeardownLock(const TeardownLock &) = delete;
    TeardownLock &operator=(const TeardownLock &) = delete;
};

/// Signals cancellation, then blocks until any in-flight generation has actually finished.
///
/// Prefer TeardownLock when the caller goes on to free something: this releases the lock
/// before returning, so it only proves generation *had* stopped.
///
/// Acquiring the generation lock is proof that generation is over, because generation holds
/// it for its whole duration. The alternative — polling a flag with a timeout — was what
/// let a 27-second image encode outlive the context it was writing into.
void stop_generation_and_wait();

/// Whether a generation is currently running. Diagnostics only: it is a snapshot, and
/// acting on it is the race that stop_generation_and_wait() exists to avoid.
bool generation_in_progress();

// ---------------------------------------------------------------------------------
// Which model is resident
// ---------------------------------------------------------------------------------

/// Whether a model loaded from `model_path` can be reused, given what is resident.
///
/// `a_model_is_loaded` says whether the bridge is holding one at all. "A model is loaded"
/// and "the requested model is loaded" are different questions, and conflating them is
/// what made loading a second vision model attach its projector to the first.
bool can_reuse_resident_model(bool a_model_is_loaded, const std::string &model_path);

/// Records the file the resident model was loaded from.
void set_resident_model_path(const std::string &model_path);

/// Forgets the resident model. Call wherever the model is freed, including on failure.
void clear_resident_model_path();

/// The file the resident model was loaded from, or empty if none.
const std::string &resident_model_path();

} // namespace llamatik
