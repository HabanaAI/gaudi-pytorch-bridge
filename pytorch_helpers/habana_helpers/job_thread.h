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
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include "logging.h"
namespace habana_helpers {
class JobThread {
 public:
  JobThread() : mJobCounter{0} {
    mTh = std::thread(&JobThread::threadFunction, this);
  }
  ~JobThread() {
    {
      std::lock_guard<std::mutex> lock(mMut);
      mFuncs.push([] { return false; });
    }
    mCondVar.notify_one();
    mTh.join();
    HABANA_ASSERT(mJobCounter == 0, "Unfinished job");
  }

  void addJob(std::function<bool()> func) {
    {
      std::lock_guard<std::mutex> lock(mMut);
      mFuncs.push(func);
      ++mJobCounter;
    }
    mCondVar.notify_one();
  }

  int jobCounter() {
    return mJobCounter;
  }

 private:
  void threadFunction() {
    while (true) {
      std::unique_lock<std::mutex> lock(mMut);
      mCondVar.wait(lock, [this] { return !mFuncs.empty(); });

      while (!mFuncs.empty()) {
        auto func = mFuncs.front();
        mFuncs.pop();
        lock.unlock();
        if (!func()) {
          // func() returns False only when it is queued by Destructor
          return;
        }
        lock.lock();
        mJobCounter--;
      }
    }
  }

  std::thread mTh;
  std::mutex mMut;
  std::atomic<int> mJobCounter;
  std::queue<std::function<bool()>> mFuncs;
  std::condition_variable mCondVar;
};

} // namespace habana_helpers
