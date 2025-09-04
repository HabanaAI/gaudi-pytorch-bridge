/**
 * Copyright (c) 2024-2025 Intel Corporation
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

#include <vector>

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include "hpu_ops/op_validator.h"
#include "stack_generator.h"
#include "utils/shared_structures.h"

namespace slrg {
template <typename Validator_t = habana::CheckNodeWithSharedLayerValidator>
class SharedLayerExecutor {
 public:
  SharedLayerExecutor() = default;
  SharedLayerExecutor(const Report& report) : report(report) {
    validated = true;
  }
  SharedLayerExecutor(IStackGenerator* generator, Validator_t* validator)
      : generator(generator), validator(validator) {}
  virtual ~SharedLayerExecutor() = default;

  virtual void validate() {
    if (validated)
      return;
    for (const auto precision_type : report_precision_types) {
      if (generator->isBlacklistedPrecisionType(precision_type)) {
        report[precision_type] = false;
      } else if (generator->isWhitelistedPrecisionType(precision_type)) {
        report[precision_type] = true;
      } else {
        const auto& stacks = generator->getStacks(precision_type);
        bool all_valid = true;
        for (const auto& stack : stacks) {
          try {
            all_valid &= checkNodeWithSharedLayer(stack);
          } catch (std::exception& e) {
            throw std::logic_error(
                "SharedLayerExecutor::validate failed for op '" +
                generator->getOpAndOverloadName() +
                "'. Possibly incorrect input quantity, signature or TORCH_CHECK inside output_meta.\nOriginal exception: " +
                e.what());
          }
        }
        report[precision_type] = all_valid;
      }
    }
    validated = true;
  }

  Report getReport() const {
    return report;
  }

 protected:
  IStackGenerator* const generator = nullptr;
  Validator_t* const validator = nullptr;
  bool validated = false;
  Report report;

  virtual bool checkNodeWithSharedLayer(const at::Stack&) const = 0;
};

template <typename Validator_t = habana::CheckNodeWithSharedLayerValidator>
class StaticSharedLayerExecutor final
    : public SharedLayerExecutor<Validator_t> {
 public:
  StaticSharedLayerExecutor(const Report& supported_types)
      : SharedLayerExecutor<Validator_t>(supported_types) {}
  virtual ~StaticSharedLayerExecutor() = default;

  void validate() override{};

 protected:
  bool checkNodeWithSharedLayer(const at::Stack&) const override {
    return true;
  }
};

template <typename Validator_t = habana::CheckNodeWithSharedLayerValidator>
class GenericSharedLayerExecutor final
    : public SharedLayerExecutor<Validator_t> {
 public:
  GenericSharedLayerExecutor(IStackGenerator* generator, Validator_t* validator)
      : SharedLayerExecutor<Validator_t>(generator, validator) {}
  virtual ~GenericSharedLayerExecutor() = default;

 protected:
  bool checkNodeWithSharedLayer(const at::Stack& stack) const override {
    const bool is_dynamic = false;
    const bool check_st_h2d = false;
    return this->validator->Validate(
        stack,
        is_dynamic,
        check_st_h2d,
        /* shared meta */ {},
        SharedLayer::DeviceId::DEVICE_ID_GAUDI2);
  }
};

template <typename Validator_t = habana::CheckNodeWithSharedLayerValidator>
class CustomSharedLayerExecutor final
    : public SharedLayerExecutor<Validator_t> {
 public:
  CustomSharedLayerExecutor(IStackGenerator* generator, Validator_t* validator)
      : SharedLayerExecutor<Validator_t>(generator, validator) {}
  virtual ~CustomSharedLayerExecutor() = default;

 protected:
  bool checkNodeWithSharedLayer(const at::Stack& stack) const override {
    const bool is_dynamic = false;
    const bool check_st_h2d = false;
    return this->validator->ValidateCustom(
        stack,
        is_dynamic,
        check_st_h2d,
        SharedLayer::DeviceId::DEVICE_ID_GAUDI2);
  }
};

} // namespace slrg
