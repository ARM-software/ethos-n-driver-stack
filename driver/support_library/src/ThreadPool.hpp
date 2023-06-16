//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ethosn
{
namespace support_library
{

/// Manages a set of background threads which execute tasks from a queue.
/// This can be used to run work in the background without blocking the current thread.
class ThreadPool
{
public:
    /// Create a thread pool with the specified number of threads (n).
    /// Specifying n=0 (i.e. no threads) will result in all tasks being run synchronously inside AddToQueue.
    /// Specifying n=-1 will result in an automatic number of threads being spawned, based
    /// on the ETHOSN_SUPPORT_LIBRARY_NUM_THREADS environment variable and the number of CPUs on the system.
    ThreadPool(int n);
    ~ThreadPool();

    /// Queues a task to be run. The `function` argument must be convertible to a std::packaged_task<void(int)>,
    /// e.g. a lambda taking an int and with no return value.
    ///   AddToQueue([](int a) { print(a); });
    template <typename T>
    std::future<void> AddToQueue(T function, int arg)
    {
        return AddToQueue(std::packaged_task<void(int)>(function), arg);
    }

private:
    std::future<void> AddToQueue(std::packaged_task<void(int)>&& function, int arg);

    // Data that we store in the queue for each task.
    struct Task
    {
        std::packaged_task<void(int)> func;
        int arg;
    };

    /// The worker threads which take tasks from the queue and execute them.
    std::vector<std::thread> m_Threads;

    /// The queue of tasks for the worker threads. This can be accessed and mutated
    /// from different threads, and so access to it needs to be guarded by `m_TaskQueueMutex`.
    std::queue<Task> m_TaskQueue;
    std::mutex m_TaskQueueMutex;

    /// Used to wake up the worker threads when new tasks are ready to be executed.
    std::condition_variable m_TaskReady;

    /// Used to tell the worker threads that they should cleanly exit, as this ThreadPool is being destroyed.
    bool m_IsShuttingDown = false;

    /// Stores which thread this ThreadPool was created on, used to avoid deadlocks.
    std::thread::id m_CreationThreadId;
};

}    // namespace support_library
}    // namespace ethosn
