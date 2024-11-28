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

#include <c10/util/intrusive_ptr.h>
#include <memory>
#include "backend/habana_device/hpu_cached_devices.h"

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
      // Destroying all process groups in order to ensure that all events
      // have been handled (all tensors connected with pending events are
      // deallocated) before Python interpreter finalization. If tensor is
      // deallocated when interpreter is down or is going down (finalizing) then
      // cPython may issue std::terminate (abort), what will be observed in DFA
      // report.
      py::object dist = py::module_::import("torch.distributed");
      py::object destroy_process_group = dist.attr("destroy_process_group");
      py::object default_pg = dist.attr("GroupMember").attr("WORLD");
      if (!default_pg.is(py::none())) {
        PT_DISTRIBUTED_DEBUG("Destroying process groups at exit");
        destroy_process_group();
      }
      // It may so happen (deepspeed being the primary example) that some
      // process groups remain after destroy_process_group. This is a problem,
      // because a process group holds a reference and prevents removal of a syn
      // device. To mitigate this, destroy all the process groups that are still
      // alive which will put them in an unattached state.
    }
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