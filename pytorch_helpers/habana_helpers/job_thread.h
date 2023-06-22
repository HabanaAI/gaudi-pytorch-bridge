/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
