/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
  virtual void MutateGraph(std::shared_ptr<torch::jit::Graph> graph) = 0;

 protected:
  // Should overwrite this with particular pass name everytime.
  std::string pass_name_ = "empty";
};

} // namespace habana
