/**
 * Copyright (c) 2021-2025 Intel Corporation
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
#include "generated/backend/logit.h"

namespace habana {
FillParamsT FillLogitParams(const at::Stack& stack, int64_t index) {
  // check if eps=None
  if (stack.at(index).isNone())
    return {};

  PARAMS_STUB(ns_LogitKernel::Params);
  params->epsilon = stack.at(index).toDouble();

  return paramsT;
}

FillParamsT FillLogitForwardParams(const at::Stack& stack) {
  // index positions for input args
  constexpr size_t epsPositionInArgList = 1;

  return FillLogitParams(stack, epsPositionInArgList);
}

FillParamsT FillLogitBackwardParams(const at::Stack& stack) {
  // index positions for input args
  constexpr size_t epsPositionInArgList = 2;

  return FillLogitParams(stack, epsPositionInArgList);
}
} // namespace habana
