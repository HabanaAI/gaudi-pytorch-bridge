/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#pragma once
#include <sys/sysinfo.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

namespace habana_helpers {

#define STD_QUEUE_CAPACITY \
  1000000 // To Do - need to review the number if std queue is used

enum QueueType { QT_LockFree, QT_WithLock, QT_Standard };

template <typename T>
class Queue {
 public:
  virtual bool emplace(const T& item) = 0;
  virtual T pop() = 0;
  virtual T& front() = 0;
  virtual bool empty() = 0;
  virtual size_t size() const = 0;
  virtual bool is_full() = 0;
  virtual size_t queue_capacity() const = 0;
  static Queue<T>* Create(QueueType type, size_t size);
  virtual ~Queue<T>() = default;
};

/**
 * @brief Used to queue messages - thread safe with lock
 *
 */

template <typename T>
// Library classes
class ThreadQueueWithLock : public Queue<T> {
 private:
  size_t m_maxSize;
  std::queue<T> m_queue{};
  std::mutex m_mutex;
  std::condition_variable m_cond;

 public:
  explicit ThreadQueueWithLock(size_t maxSize) {
    m_maxSize = maxSize;
  }
  ~ThreadQueueWithLock() {
    while (!m_queue.empty()) {
      m_queue.pop();
    }
    m_maxSize = 0;
  }
  size_t queue_capacity() const {
    return m_maxSize;
  }

  size_t size() const {
    return m_queue.size();
  }

  bool empty() {
    return m_queue.empty();
  }

  bool is_full() {
    return (size() >= queue_capacity());
  }

  T pop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cond.wait(lock, [this] { return !m_queue.empty(); });
    T item = std::move(m_queue.front());
    m_queue.pop();
    m_cond.notify_one();
    return item;
  }

  bool emplace(const T& data) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cond.wait(lock, [this] { return (m_queue.size() < m_maxSize); });
    m_queue.push(data);
    lock.unlock();
    m_cond.notify_one();
    return true;
  }

  T& front() {
    return m_queue.front();
  }
};

/**
 * @brief Used to queue messages between threads - atomic queue
 *
 */
template <typename T>
class ThreadQueueLockFree : public Queue<T> {
 private:
  struct Node {
    T data;
    std::atomic<size_t> tail{};
    std::atomic<size_t> head{};
  };

  size_t capacityMask;
  size_t capacityQueue{};
  std::atomic<size_t> q_tail{};
  std::atomic<size_t> q_head{};
  Node* queue;
  Node* node = nullptr;
  Node anode;

 public:
  explicit ThreadQueueLockFree(size_t capacity) {
    capacityMask = capacity - 1;
    for (size_t i = 1; i <= sizeof(void*) * 4; i <<= 1)
      capacityMask |= capacityMask >> i;
    capacityQueue = capacityMask + 1;

    // element size as 32 for the args list
    size_t modif_size = sizeof(Node) * 32;
    queue = (Node*)new char[modif_size * capacityQueue];

    for (size_t i = 0; i < capacityQueue; ++i) {
      queue[i].tail.store(i, std::memory_order_relaxed);
      queue[i].head.store(-1, std::memory_order_relaxed);
    }

    q_tail.store(0, std::memory_order_relaxed);
    q_head.store(0, std::memory_order_relaxed);
  }

  ~ThreadQueueLockFree() {
    for (size_t i = q_head; i != q_tail; ++i)
      (&queue[i & capacityMask].data)->~T();

    delete[](char*) queue;
  }

  size_t queue_capacity() const {
    return capacityQueue;
  }

  size_t size() const {
    size_t head = q_head.load(std::memory_order_acquire);
    return q_tail.load(std::memory_order_relaxed) - head;
  }

  bool empty() {
    return size() == 0;
  }

  bool is_full() {
    return (size() >= queue_capacity());
  }

  bool emplace(const T& data) {
    HABANA_ASSERT(!is_full() && "Message queue is full");
    node = &queue[q_tail & capacityMask];
    node->head.store(q_tail, std::memory_order_relaxed);
    q_tail.store(q_tail + 1, std::memory_order_relaxed);

    // placement new and no allocation
    new (&node->data) T(data);
    return true;
  }

  T pop() {
    HABANA_ASSERT(!empty() && "Message queue is empty");
    node = &queue[q_head & capacityMask];
    q_head.store(q_head + 1, std::memory_order_relaxed);
    node->tail.store(q_head + capacityQueue, std::memory_order_release);
    return node->data;
  }

  T& front() {
    Node* node = nullptr;
    node = &queue[q_head & capacityMask];
    return node->data;
  }
};

/**
 * @brief Used to queue messages with standard queue
 *
 */

template <typename T>
// Library classes
class StdQueue : public Queue<T> {
 public:
  std::queue<T> m_queue;

  bool empty() {
    return m_queue.empty();
  }

  T pop() {
    T item = std::move(m_queue.front());
    m_queue.pop();
    return item;
  }

  bool emplace(const T& data) {
    m_queue.push(data);
    return true;
  }

  size_t size() const {
    return m_queue.size();
  }

  T& front() {
    return m_queue.front();
  }

  size_t queue_capacity() const {
    // To Do
    return STD_QUEUE_CAPACITY;
  }

  bool is_full() {
    // To Do
    return false;
  }
};

template <typename T>
Queue<T>* Queue<T>::Create(QueueType type, size_t size) {
  if (type == QT_LockFree) {
    return new ThreadQueueLockFree<T>{size};
  } else if (type == QT_WithLock) {
    return new ThreadQueueWithLock<T>{size};
  } else if (type == QT_Standard) {
    return new StdQueue<T>;
  } else {
    return NULL;
  }
}

} // namespace habana_helpers
