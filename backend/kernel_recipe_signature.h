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
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/script.h>
#include <functional>
#include <string>
#include <vector>
#include "backend/helpers/tensor_utils.h"

namespace habana_helpers {

struct RecipeSignature {
  RecipeSignature(
      bool with_grad,
      torch::jit::Stack inputs,
      std::vector<std::string> nodeTypes,
      bool in_place = false,
      bool outOp = false)
      : nodeTypes_(nodeTypes), cas_(with_grad, inputs), hash_(cas_.hashCode()) {
    // flatten tensorlist and insert tensors into inputs stack
    auto num_inputs = inputs.size();
    for (unsigned i = 0; i < num_inputs; i++) {
      if (inputs[i].isTensorList()) {
        auto tlist = inputs[i].toTensorList();
        auto tlsize = tlist.size();
        for (unsigned j = 0; j < tlsize; j++) {
          inputs.push_back(tlist.get(j));
        }
      }
    }

    // compute hash on inputs
    cas_ = torch::jit::CompleteArgumentSpec(with_grad, inputs);
    hash_ = cas_.hashCode();

    // calculate operator cache
    for (auto name : nodeTypes) {
      hash_ =
          at::hash_combine(hash_, std::hash<std::string>{}(std::string(name)));
    }

    // if inplace is true add the value, normal and inplace operation
    // may have the same node and inputs. since the kernel generated
    // are different, need to do this.
    hash_ = at::hash_combine(hash_, in_place);
    hash_ = at::hash_combine(hash_, outOp);
    hash_ = hash_combine_scalars(hash_, inputs);
  }

  bool operator==(const RecipeSignature& rv) const {
    return cas_ == rv.cas_ && nodeTypes_ == rv.nodeTypes_;
  }
  bool operator!=(const RecipeSignature& rv) const {
    return !(*this == rv);
  }

  size_t hash() const {
    return hash_;
  }

 private:
  std::vector<std::string> nodeTypes_;
  torch::jit::CompleteArgumentSpec cas_;
  size_t hash_;
};
}; // namespace habana_helpers
