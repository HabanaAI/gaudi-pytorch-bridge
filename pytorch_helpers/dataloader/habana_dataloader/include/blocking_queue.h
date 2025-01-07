/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
// clang-format off

#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

template <typename T>
class BlockingQueue
{
public:
    BlockingQueue(unsigned maxSize) {
        m_maxSize = maxSize;
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return !m_queue.empty(); });
        T item = std::move(m_queue.front());
        m_queue.pop();
        m_cond.notify_one();
        return item;
    }

    void pop(T& item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return !m_queue.empty(); });
        item = std::move(m_queue.front());
        m_queue.pop();
        m_cond.notify_one();
    }

    void top(T& item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return !m_queue.empty(); });
        item = m_queue.front();
    }

    void push(const T& item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return (m_queue.size() < m_maxSize); });
        m_queue.push(item);
        lock.unlock();
        m_cond.notify_one();
    }

    void push(T&& item)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [this] { return (m_queue.size() < m_maxSize); });
        m_queue.push(std::move(item));
        lock.unlock();
        m_cond.notify_one();
    }

    void clear()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        std::queue<T>                empty;
        std::swap(m_queue, empty);
        lock.unlock();
        m_cond.notify_one();
    }

private:
    unsigned                m_maxSize;
    std::queue<T>           m_queue;
    std::mutex              m_mutex;
    std::condition_variable m_cond;
};
