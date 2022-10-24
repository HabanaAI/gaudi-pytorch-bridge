/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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

#include <atomic>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace habana_lazy {

class AccThreadPool {
 public:
  using AccTask = std::function<void()>;

  AccThreadPool();
  ~AccThreadPool();

  bool inThreadPool() const;
  void run(AccTask func);
  void waitWorkComplete();

 private:
  std::queue<AccTask> tasks_;
  std::vector<std::thread> threads_;
  mutable std::mutex mutex_;
  std::atomic_bool running_;
  std::atomic<std::size_t> task_count_;

  void main_loop();
};

} // namespace habana_lazy