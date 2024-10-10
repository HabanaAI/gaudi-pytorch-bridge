/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include <memory>

#include <c10/util/intrusive_ptr.h>

namespace c10d {

template <class PG_TYPE>
class ProcessGroupHCCLRegistry {
 public:
  using intrptr_t = c10::intrusive_ptr<PG_TYPE>;
  using weakptr_t = c10::weak_intrusive_ptr<PG_TYPE>;

  static ProcessGroupHCCLRegistry<PG_TYPE>& instance() {
    std::call_once(init_once_flag_, []() {
      instance_.reset(new ProcessGroupHCCLRegistry<PG_TYPE>());
      habana::HPURegistrar::get_hpu_registrar()
          .register_process_group_finalizer(habana::CallFinally([]() {
            PT_DISTRIBUTED_DEBUG("ProcessGroupHCCLRegistry finalizer");
            instance_.reset();
          }));
    });

    return *instance_;
  }

  static intrptr_t create(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      std::string group_name) {
    auto r = c10::make_intrusive<PG_TYPE>(store, rank, size, group_name);
    ProcessGroupHCCLRegistry<PG_TYPE>::instance().insert(r);

    return r;
  }

  ~ProcessGroupHCCLRegistry() {
    cleanup();
  }

 private:
  std::vector<weakptr_t> groups_;
  std::mutex groups_mutex_;
  static std::once_flag init_once_flag_;
  static std::unique_ptr<ProcessGroupHCCLRegistry<PG_TYPE>> instance_;

  void cleanup() {
    PT_DISTRIBUTED_DEBUG("Clearing distributed process groups");
    {
      std::unique_lock<std::mutex> lock{groups_mutex_};

      auto it = groups_.rbegin();
      while (it != groups_.rend()) {
        auto wp{it->lock()};
        if (wp) {
          PT_DISTRIBUTED_DEBUG(
              "There are active references to process group. Destroying.");
          wp->destroy();
          it++;
        } else {
          PT_DISTRIBUTED_DEBUG(
              "All references to process group have been removed. Removing weak reference.");
          // When use_count of target pointed by intrusive_ptr reaches 0 in most
          // cases the target memory is freed. However (for some reason) when
          // weak reference count pointing to target is > 1 then memory is not
          // freed. Therefore, we need to remove weak reference stored in
          // groups_ to ensure that process group object is deallocated.
          auto new_fwd_iter = groups_.erase(it.base() - 1);
          it = std::reverse_iterator(new_fwd_iter);
        }
      }
    }

    groups_.clear();
  }

  void insert(intrptr_t& new_pg) {
    std::unique_lock<std::mutex> lock{groups_mutex_};
    weakptr_t new_pg_weak{new_pg};
    groups_.push_back(std::move(new_pg_weak));
  }
};

template <class T>
std::once_flag ProcessGroupHCCLRegistry<T>::init_once_flag_;
template <class T>
std::unique_ptr<ProcessGroupHCCLRegistry<T>>
    ProcessGroupHCCLRegistry<T>::instance_;

} // namespace c10d