// Tests for the generation lifecycle rules shared by the Android and iOS bridges.
//
// These cover the three bugs that shipped twice each, once per platform: a model freed
// while an encode was still running, a resident model reused for a different one, and a
// stale cancellation killing the next session.
//
// No llama.cpp, no model, no device — engine_state depends on nothing but the standard
// library, which is the point of having extracted it.

#include "engine_state.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

using namespace llamatik;
using namespace std::chrono_literals;

static int g_failures = 0;

static void check(bool condition, const char *what) {
    if (condition) {
        std::printf("  ok   %s\n", what);
    } else {
        std::printf("  FAIL %s\n", what);
        ++g_failures;
    }
}

// Regression: freeing the model used to poll a flag for 5s and then free regardless. An
// image encode runs far longer than that, finished after the free, and wrote into a
// deallocated context — SIGSEGV, tens of seconds after the apparent cause.
static void waits_for_generation_to_actually_finish() {
    std::printf("waits_for_generation_to_actually_finish\n");

    std::atomic<bool> generation_done{false};
    std::atomic<bool> waiter_returned{false};

    std::thread generating([&] {
        GenerationLock lock;
        // Stands in for a long image encode: uninterruptible, and much longer than any
        // timeout someone might be tempted to use here.
        std::this_thread::sleep_for(300ms);
        generation_done.store(true);
    });

    std::this_thread::sleep_for(50ms); // let the generation take the lock first

    std::thread freeing([&] {
        stop_generation_and_wait();
        // If this observes generation_done == false, the model would have been freed
        // under a running generation.
        check(generation_done.load(), "generation finished before the wait returned");
        waiter_returned.store(true);
    });

    generating.join();
    freeing.join();
    check(waiter_returned.load(), "the wait returned rather than deadlocking");
}

// Regression: llama_vision_init reused whatever model was resident, so loading a second
// model attached its projector to the first and failed until the process restarted.
static void reuses_a_model_only_when_it_is_the_one_requested() {
    std::printf("reuses_a_model_only_when_it_is_the_one_requested\n");

    clear_resident_model_path();
    check(!can_reuse_resident_model(false, "/models/a.gguf"), "nothing resident means no reuse");

    set_resident_model_path("/models/a.gguf");
    check(can_reuse_resident_model(true, "/models/a.gguf"), "same path reuses");
    check(!can_reuse_resident_model(true, "/models/b.gguf"), "different path does not reuse");

    // A pointer without a recorded path is the pre-fix state: something is loaded but
    // nobody knows what. Reusing it is the bug.
    clear_resident_model_path();
    check(!can_reuse_resident_model(true, "/models/a.gguf"), "unknown resident model is not reused");

    // Freeing must forget the path, or the next load reuses a model that is gone.
    set_resident_model_path("/models/a.gguf");
    clear_resident_model_path();
    check(resident_model_path().empty(), "clearing forgets the path");
}

// Regression: shutdown sets cancel-everything. If that outlives the shutdown, the next
// generation is cancelled before it emits a token and appears to do nothing at all.
static void a_stale_cancellation_does_not_kill_the_next_session() {
    std::printf("a_stale_cancellation_does_not_kill_the_next_session\n");

    const uint64_t first = begin_session();
    cancel_session(first);
    check(should_cancel_session(first), "a cancelled session is cancelled");

    const uint64_t second = begin_session();
    check(second != first, "each session gets its own id");
    check(!should_cancel_session(second), "the next session starts uncancelled");

    cancel_all_sessions();
    check(should_cancel_session(second), "cancel-all reaches a running session");

    const uint64_t third = begin_session();
    check(!should_cancel_session(third), "cancel-all does not leak into the next session");
}

// Cancellation names one session, so a slow generation finishing late cannot cancel the
// generation that replaced it.
static void cancelling_one_session_leaves_others_alone() {
    std::printf("cancelling_one_session_leaves_others_alone\n");

    const uint64_t current = begin_session();
    cancel_session(current - 1); // a stale request for a session already gone
    check(!should_cancel_session(current), "a stale session id does not cancel the current one");
}

// Cancelling "whatever is running" has to name the running session, not a constant, or a
// late cancel would kill whichever generation happened to be running when it arrived.
static void cancelling_the_current_session_targets_it_by_id() {
    std::printf("cancelling_the_current_session_targets_it_by_id\n");

    const uint64_t running = begin_session();
    check(current_session() == running, "current_session reports the running session");

    cancel_session(current_session());
    check(should_cancel_session(running), "cancelling the current session cancels it");
    check(pending_cancel_session() == running, "the pending cancel names that session");

    const uint64_t next = begin_session();
    check(!should_cancel_session(next), "the cancel does not carry into the next session");
}

// Teardown must hold the lock across the free, not just wait and let go: releasing first
// lets a generation start on a context that is about to be freed.
static void teardown_holds_the_lock_while_it_runs() {
    std::printf("teardown_holds_the_lock_while_it_runs\n");

    std::atomic<bool> generation_started{false};

    {
        TeardownLock teardown;
        std::thread contender([&] {
            GenerationLock lock;      // must block until the teardown scope ends
            generation_started.store(true);
        });
        std::this_thread::sleep_for(80ms);
        check(!generation_started.load(), "no generation can start while teardown holds the lock");
        // Releasing here lets the contender through.
        contender.detach();
    }

    std::this_thread::sleep_for(80ms);
    check(generation_started.load(), "generation proceeds once teardown releases");
}


// A second model chosen in the chooser must supersede the first immediately. Without this
// the blocking native load ran to completion and the new selection queued behind it.
static void a_newer_load_supersedes_the_one_running() {
    std::printf("a_newer_load_supersedes_the_one_running\n");

    const uint64_t first = begin_load();
    check(!load_superseded(first), "the only load in flight is not superseded");

    const uint64_t second = begin_load();
    check(load_superseded(first), "the first load is superseded once a second starts");
    check(!load_superseded(second), "the newest load keeps running");
    check(first != second, "each load gets its own id");
}

// Regression, by analogy with the generation bug: a shutdown cancel that stays pending
// kills the next load, which looks like model selection silently doing nothing.
static void a_shutdown_cancel_does_not_kill_the_next_load() {
    std::printf("a_shutdown_cancel_does_not_kill_the_next_load\n");

    const uint64_t during_shutdown = begin_load();
    cancel_all_loads();
    check(load_superseded(during_shutdown), "cancel_all_loads stops the load in flight");

    const uint64_t after_restart = begin_load();
    check(!load_superseded(after_restart), "a load started afterwards runs normally");
}

// The abort hook runs on the loading thread while the chooser starts another load on the
// main thread, so supersession has to be visible across threads.
static void supersession_is_visible_to_the_loading_thread() {
    std::printf("supersession_is_visible_to_the_loading_thread\n");

    const uint64_t loading = begin_load();
    std::atomic<bool> observed_cancel{false};

    std::thread loader([&] {
        for (int i = 0; i < 200; ++i) {
            if (load_superseded(loading)) {
                observed_cancel.store(true);
                return;
            }
            std::this_thread::sleep_for(1ms);
        }
    });

    std::this_thread::sleep_for(10ms);
    begin_load();
    loader.join();

    check(observed_cancel.load(), "the loading thread sees it has been superseded");
}

int main() {
    waits_for_generation_to_actually_finish();
    reuses_a_model_only_when_it_is_the_one_requested();
    a_stale_cancellation_does_not_kill_the_next_session();
    cancelling_one_session_leaves_others_alone();
    cancelling_the_current_session_targets_it_by_id();
    teardown_holds_the_lock_while_it_runs();
    a_newer_load_supersedes_the_one_running();
    a_shutdown_cancel_does_not_kill_the_next_load();
    supersession_is_visible_to_the_loading_thread();

    if (g_failures == 0) {
        std::printf("\nengine_state: all checks passed\n");
        return 0;
    }
    std::printf("\nengine_state: %d check(s) FAILED\n", g_failures);
    return 1;
}
