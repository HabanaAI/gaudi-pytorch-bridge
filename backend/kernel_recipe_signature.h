/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
