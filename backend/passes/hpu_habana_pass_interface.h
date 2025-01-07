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

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_set>

#include <torch/csrc/jit/ir/ir.h>

namespace habana {

// Base class
template <typename R>
class JITGraphPass {
 public:
  // pure virtual function providing interface framework.
  virtual std::unique_ptr<R> VisitGraph(
      const std::shared_ptr<torch::jit::Graph> graph) = 0;

 protected:
  // Should overwrite this with particular pass name everytime.
  std::string pass_name_ = "empty";
};

} // namespace habana
