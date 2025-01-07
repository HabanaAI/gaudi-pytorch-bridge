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

#include "generated/lazy/addcdiv.h"
#include "generated/lazy/addcmul.h"
namespace habana {

static void convert_scalar_val_to_tensor(at::Stack& inputs) {
  auto self = inputs.at(0).toTensor();
  auto value = inputs.at(3).toScalar();
  at::Tensor valueTensor;
  if (!value.equal(1))
    valueTensor =
        habana_lazy::get_tensor_for_scalar(value.to<double>(), self.options());

  c10::optional<at::Tensor> valueTensorOpt = c10::make_optional(valueTensor);
  inputs.at(3) = valueTensorOpt;
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, AddCOpFE, at::Tensor&) {
  convert_scalar_val_to_tensor(get_inputs());
}

HPU_OP_FRONTEND_CUSTOM_CTOR_ONLY(habana_lazy::LazyOp, AddCOpFE, at::Tensor) {
  convert_scalar_val_to_tensor(get_inputs());
}

} // namespace habana
