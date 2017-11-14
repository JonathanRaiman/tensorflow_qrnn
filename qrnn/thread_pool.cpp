#include "thread_pool.h"

#include <chrono>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cassert>
#include <deque>
#include <vector>

class ThreadPool {
    private:
        typedef std::chrono::duration<double> Duration;
        static __thread bool in_thread_pool;
        // c++ assigns random id to each thread. This is not a thread_id
        // it's a number inside this thread pool.
        static __thread int thread_number;

        bool should_terminate;
        std::mutex queue_mutex;
        std::condition_variable is_idle;
        int active_count;

        std::deque<std::function<void()> > work;
        std::vector<std::thread> pool;
        Duration between_queue_checks;

        void thread_body(int _thread_id);
    public:
        // Creates a thread pool composed of num_threads threads.
        // threads are started immediately and exit only once ThreadPool
        // goes out of scope. Threads periodically check for new work
        // and the frequency of those checks is at minimum between_queue_checks
        // (it can be higher due to thread scheduling).
        ThreadPool(int num_threads, Duration between_queue_checks=std::chrono::milliseconds(1));

        // Run a function on a thread in pool.
        void run(std::function<void()> f);

        // Wait until queue is empty and all the threads have finished working.
        // If timeout is specified function waits at most timeout until the
        // threads are idle. If they indeed become idle returns true.
        bool wait_until_idle(Duration timeout);
        bool wait_until_idle();

        // Retruns true if all the work is done.
        bool idle() const;
        // Return number of active busy workers.
        int active_workers();
        ~ThreadPool();
};

__thread bool ThreadPool::in_thread_pool = false;
__thread int ThreadPool::thread_number = -1;

ThreadPool::ThreadPool(int num_threads, Duration between_queue_checks) :
        between_queue_checks(between_queue_checks),
        should_terminate(false),
        active_count(0) {
    // Thread pool inception is not supported at this time.
    assert(!in_thread_pool);

    ThreadPool::between_queue_checks = between_queue_checks;
    for (int thread_number = 0; thread_number < num_threads; ++thread_number) {
        pool.emplace_back(&ThreadPool::thread_body, this, thread_number);
    }
}

void ThreadPool::thread_body(int _thread_id) {
    in_thread_pool = true;
    thread_number = _thread_id;
    bool am_i_active = false;

    while (true) {
        std::function<void()> f;
        {
            std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
            bool was_i_active = am_i_active;
            if (should_terminate && work.empty())
                break;
            if (!work.empty()) {
                am_i_active = true;
                f = work.front();
                work.pop_front();
            } else {
                am_i_active = false;
            }

            if (am_i_active != was_i_active) {
                active_count += am_i_active ? 1 : -1;
                if (active_count == 0) {
                    // number of workers decrease so maybe all are idle
                    is_idle.notify_all();
                }
            }
        }
        // Function defines implicit conversion to bool
        // which is true only if call target was set.
        if (static_cast<bool>(f)) {
            f();
        } else {
            std::this_thread::sleep_for(between_queue_checks);
        }
        std::this_thread::yield();
    }
}

int ThreadPool::active_workers() {
    std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
    return active_count;
}

bool ThreadPool::wait_until_idle(Duration timeout) {
    std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
    is_idle.wait_for(lock, timeout, [this]{
        return active_count == 0 && work.empty();
    });
    return idle();
}

bool ThreadPool::wait_until_idle() {
    int retries = 3;
    while (retries--) {
        try {
            std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
            is_idle.wait(lock, [this]{
                return active_count == 0 && work.empty();
            });
            return idle();
        } catch (...) {}
    }
    throw std::runtime_error(
        "exceeded retries when waiting until idle."
    );
    return false;
}

bool ThreadPool::idle() const {
    return active_count == 0 && work.empty();
}

void ThreadPool::run(std::function<void()> f) {
    int retries = 3;
    while (retries--) {
        try {
            std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
            work.push_back(f);
            return;
        } catch (...) {}
    }
    throw std::runtime_error(
        "exceeded retries when trying to run operation on thread pool."
    );
}

ThreadPool::~ThreadPool() {
    // Terminates thread pool making sure that all the work
    // is completed.
    should_terminate = true;
    for (auto& t : pool)
        t.join();
}


void ParallelFor(int max_parallelism, int num_threads, int cost_per_unit, int total,
                 std::function<void(int, int)> work) {
    cost_per_unit = std::max(1, cost_per_unit);
    // We shard [0, total) into "num_shards" shards.
    //   1 <= num_shards <= num worker threads
    //
    // If total * cost_per_unit is small, it is not worth shard too
    // much. Let us assume each cost unit is 1ns, kMinCostPerShard=10000
    // is 10us.
    static const int kMinCostPerShard = 10000;
    const int num_shards =
        std::max<int>(1, std::min(static_cast<int>(max_parallelism),
                                  total * cost_per_unit / kMinCostPerShard));
    // Each shard contains up to "block_size" units. [0, total) is sharded
    // into:
    //   [0, block_size), [block_size, 2*block_size), ...
    // The 1st shard is done by the caller thread and the other shards
    // are dispatched to the worker threads. The last shard may be smaller than
    // block_size.
    const int block_size = (total + num_shards - 1) / num_shards;
    if (block_size <= 0) {
        throw std::runtime_error("block_size must be > 0.");
    }
    if (block_size >= total) {
       work(0, total);
       return;
    }
    const int num_shards_used = (total + block_size - 1) / block_size;
    ThreadPool pool(num_threads);
    for (int start = block_size; start < total; start += block_size) {
        auto limit = std::min(start + block_size, total);
        pool.run([&work, start, limit]() {
            work(start, limit);        // Compute the shard.
        });
    }
    // Inline execute the 1st shard.
    work(0, std::min(block_size, total));
    pool.wait_until_idle();
}


void Shard(int max_parallelism, int num_threads, int total,
           int cost_per_unit, std::function<void(int, int)> work) {
    if (total < 0) {
        std::runtime_error("total must be > 0.");
    }
    if (total == 0) {
        return;
    }
    if (max_parallelism <= 1) {
        // Just inline the whole work since we only have 1 thread (core).
        work(0, total);
        return;
    }
    ParallelFor(max_parallelism, num_threads,
                std::max(1, cost_per_unit), total, work);
    return;
}
