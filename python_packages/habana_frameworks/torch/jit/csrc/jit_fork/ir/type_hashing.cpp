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
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "jit_fork/ir/type_hashing.h"

#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/hash.h>

#include "jit_fork/ir/ir.h"

namespace habana_torch::jit {

namespace {
size_t hashType(const Type& type) {
  if (auto named_type = type.castRaw<ClassType>()) {
    return get_hash(named_type->name().value());
  }
  size_t hash = 0;
  for (const auto& containedType : type.containedTypes()) {
    hash = at::hash_combine(hash, hashType(*containedType));
  }
  at::hash_combine(hash, get_hash(type.kind()));
  return hash;
}
} // namespace

size_t HashType::operator()(const TypePtr& type) const {
  return hashType(*type);
}

size_t HashType::operator()(const c10::ConstTypePtr& type) const {
  return hashType(*type);
}

bool EqualType::operator()(const TypePtr& a, const TypePtr& b) const {
  return *a == *b;
}

bool EqualType::operator()(
    const c10::ConstTypePtr& a,
    const c10::ConstTypePtr& b) const {
  return *a == *b;
}

} // namespace habana_torch::jit
