//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "ThreadPool.hpp"

#include "Utils.hpp"

#include <string>

namespace ethosn
{
namespace support_library
{

ThreadPool::ThreadPool(int n)
    : m_CreationThreadId(std::this_thread::get_id())
{
    if (n == -1)
    {
        // Automatically determine a number of threads to use.
        // First try the environment variable, and if that hasn't been provided then base it on the
        // number of CPU cores.
        char* envVar = std::getenv("ETHOSN_SUPPORT_LIBRARY_NUM_THREADS");
        if (envVar != nullptr && strlen(envVar) > 0)
        {
            try
            {
                n = std::stoi(envVar);
            }
            catch (std::logic_error&)
            {
                n = 0;
            }
        }
        else
        {
            // Half the number of CPU cores to avoid taking all the resources.
            n = utils::DivRoundUp(std::thread::hardware_concurrency(), 2);
        }
    }

    // Spawn the worker threads. Initially they will block waiting for new tasks.
    auto workerThread = [this]() {
        while (true)
        {
            // Wait for new tasks or shutdown. These are both signalled by the m_TaskReady condition variable.
            std::unique_lock<std::mutex> lock(m_TaskQueueMutex);
            m_TaskReady.wait(lock, [&]() { return m_TaskQueue.size() > 0 || m_IsShuttingDown; });

            if (m_IsShuttingDown)
            {
                break;
            }

            // Take the next task from the queue and execute it
            Task ourTask = std::move(m_TaskQueue.front());
            m_TaskQueue.pop();

            // We're finished with the task queue now, so it's important that we
            // unlock it for other threads before we execute the (potentially long-running task).
            lock.unlock();

            // Run it on this thread
            ourTask.func(ourTask.arg);
        }
    };

    for (int i = 0; i < n; i++)
    {
        m_Threads.push_back(std::thread(workerThread));
    }
}

std::future<void> ThreadPool::AddToQueue(std::packaged_task<void(int)>&& function, int arg)
{
    std::future<void> future = function.get_future();

    // If a task is queued from one of the worker threads (rather than the thread on which this ThreadPool
    // was created), then it could lead to a deadlock (all the worker threads are running tasks which
    // have queued a new task and are then waiting for it, but these new tasks can't run because
    // all the worker threads are busy waiting).
    // To avoid this automatically, we run tasks spawned from worker threads synchronously instead.
    // We also support disabling multithreading by setting n=0
    if (m_Threads.empty() || std::this_thread::get_id() != m_CreationThreadId)
    {
        function(arg);
    }
    else
    {
        // Add the task to the queue
        {
            std::lock_guard<std::mutex> guard(m_TaskQueueMutex);
            m_TaskQueue.push(Task{ std::move(function), arg });
        }
        // Signal one worker thread to wake up to execute the task.
        m_TaskReady.notify_one();
    }

    return future;
}

ThreadPool::~ThreadPool()
{
    // Inform all the threads to exit cleanly.
    // Note that the mutex must be locked to ensure that the change is visible
    // to the threads (see condition_variable docs)
    {
        std::lock_guard<std::mutex> guard(m_TaskQueueMutex);
        m_IsShuttingDown = true;
    }
    m_TaskReady.notify_all();

    // Wait for all threads to stop
    for (auto&& t : m_Threads)
    {
        t.join();
    }
}

}    // namespace support_library
}    // namespace ethosn
