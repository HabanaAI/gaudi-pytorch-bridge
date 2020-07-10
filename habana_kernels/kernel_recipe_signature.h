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
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/script.h>
#include <functional>
#include <string>
#include <vector>

namespace habana_helpers {

struct RecipeSignature {
  RecipeSignature(
      bool with_grad,
      torch::jit::Stack inputs,
      std::vector<std::string> nodeTypes,
      bool in_place = false,
      bool outOp = false)
      : nodeTypes_(nodeTypes), cas_(with_grad, inputs), hash_(cas_.hashCode()) {
    // calcualte operator cache
    for (auto name : nodeTypes) {
      hash_ = torch::hash_combine(
          hash_, std::hash<std::string>{}(std::string(name)));
    }

    // if inplace is true add the value, normal and inplace operation
    // may have the same node and inputs. since the kernel generated
    // are different, need to do this.
    hash_ = torch::hash_combine(hash_, in_place);
    hash_ = torch::hash_combine(hash_, outOp);
    const int32_t num_inputs = inputs.size();
    for (int32_t i = 0; i < num_inputs; i++) {
      if (!inputs[i].isTensor()) {
        if (inputs[i].isInt()) {
          int val = inputs[i].toInt();
          std::hash<int> valhash;
          hash_ = torch::hash_combine(hash_, valhash(val));
        } else if (inputs[i].isBool()) {
          bool val = inputs[i].toBool();
          hash_ = torch::hash_combine(hash_, val);
        } else if (inputs[i].isDouble()) {
          double val = inputs[i].toDouble();
          std::hash<double> valhash;
          hash_ = torch::hash_combine(hash_, valhash(val));
        } else if (inputs[i].isList()) {
          auto vlist = inputs[i].toListRef();
          for (auto& v : vlist) {
            if (v.isInt()) {
              int val = v.toInt();
              std::hash<int> valhash;
              hash_ = torch::hash_combine(hash_, valhash(val));
            } else if (v.isBool()) {
              hash_ = torch::hash_combine(hash_, v.toBool());
            } else if (v.isDouble()) {
              double val = v.toDouble();
              std::hash<double> valhash;
              hash_ = torch::hash_combine(hash_, valhash(val));
            }
          }
        }
      }
    }
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
