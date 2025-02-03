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

#include <ATen/core/TensorBody.h>
#include <absl/functional/any_invocable.h>
#include "backend/backend_meta.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "eager_tensor.h"
#include "habana_eager/eager_context.h"

namespace habana::eager {

enum class ThreadType {
  LOWERING,
  COMPILE,
  EXECUTE,
};

template <typename C>
class PipeliningTaskAllThreads {
 public:
  // Constructor that accepts an arbitrary number of arguments and two functions
  template <typename F1, typename F2, typename F3>
  PipeliningTaskAllThreads(
      C&& c,
      F1&& func_lowering,
      F2&& func_compile,
      F3&& func_execute)
      : data(std::forward<C>(c)),
        func_lowering_(std::forward<F1>(func_lowering)),
        func_compile_(std::forward<F2>(func_compile)),
        func_execute_(std::forward<F3>(func_execute)) {}

  void LoweringCall() {
    if (func_lowering_)
      func_lowering_(data);
  }

  void CompileCall() {
    func_compile_(data);
  }

  void ExecuteCall() {
    func_execute_(data);
  }

  constexpr bool IsCompileNeeded() {
    return true;
  }
  constexpr bool IsExecuteNeeded() {
    return true;
  }

 private:
  C data;
  absl::AnyInvocable<void(C&)> func_lowering_;
  absl::AnyInvocable<void(C&)> func_compile_;
  absl::AnyInvocable<void(C&)> func_execute_;
};

// Deduction guide
template <typename F1, typename F2, typename F3, typename C>
PipeliningTaskAllThreads(C&&, F1&&, F2&&, F3&&) -> PipeliningTaskAllThreads<C>;

template <ThreadType thread_type>
class PipeliningTask {
  absl::AnyInvocable<void()> task_;

 public:
  PipeliningTask(absl::AnyInvocable<void()> task) : task_(std::move(task)) {
    HABANA_ASSERT(task_);
  }

  void LoweringCall() {
    if constexpr (thread_type == ThreadType::LOWERING)
      task_();
  }
  void CompileCall() {
    if constexpr (thread_type == ThreadType::COMPILE)
      task_();
  }
  void ExecuteCall() {
    if constexpr (thread_type == ThreadType::EXECUTE)
      task_();
  }
  constexpr bool IsCompileNeeded() {
    return thread_type == ThreadType::COMPILE ||
        thread_type == ThreadType::EXECUTE;
  }
  constexpr bool IsExecuteNeeded() {
    return thread_type == ThreadType::EXECUTE;
  }
};

template <typename T>
class PipeliningExecutor {
 public:
  static void LoweringStage(T&& pipe_task) {
    pipe_task.LoweringCall();

    if (!GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
      habana::HPUDeviceContext::compile_thread().waitWorkComplete();
      habana::HPUDeviceContext::execute_thread().waitWorkComplete();
      pipe_task.CompileCall();
      pipe_task.ExecuteCall();
      return;
    }

    if (pipe_task.IsCompileNeeded())
      habana::HPUDeviceContext::compile_thread().enqueue(
          PipeliningExecutor::CompileStage, std::move(pipe_task));
  }

  static void CompileStage(T&& pipe_task) {
    pipe_task.CompileCall();
    if (pipe_task.IsExecuteNeeded())
      habana::HPUDeviceContext::execute_thread().enqueue(
          PipeliningExecutor::ExecuteStage, std::move(pipe_task));
  }

  static void ExecuteStage(T&& pipe_task) {
    pipe_task.ExecuteCall();
  }
};

template <ThreadType thread_type>
void PipelineTask(absl::AnyInvocable<void()>&& task) {
  HPUDeviceContext::lowering_thread().enqueue(
      PipeliningExecutor<PipeliningTask<thread_type>>::LoweringStage,
      PipeliningTask<thread_type>(std::move(task)));
}

template <typename F1, typename F2, typename F3, typename C>
void PipelineTaskAllThreads(
    C&& c,
    F1&& func_lowering,
    F2&& func_compile,
    F3&& func_execute) {
  PipeliningTaskAllThreads task(
      std::move(c),
      std::move(func_lowering),
      std::move(func_compile),
      std::move(func_execute));

  HPUDeviceContext::lowering_thread().enqueue(
      PipeliningExecutor<PipeliningTaskAllThreads<C>>::LoweringStage,
      std::move(task));
}

template <typename F1, typename F2, typename C>
void PipelineTaskLowering(C&& c, F1&& func_compile, F2&& func_execute) {
  PipeliningTaskAllThreads task(
      std::move(c), nullptr, std::move(func_compile), std::move(func_execute));

  PipeliningExecutor<PipeliningTaskAllThreads<C>>::LoweringStage(
      std::move(task));
}

inline void PipelineOrExecuteTask(absl::AnyInvocable<void()>&& task) {
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE)) {
    PipelineTask<ThreadType::LOWERING>(std::move(task));
  } else {
    task();
  }
}

} // namespace habana::eager
