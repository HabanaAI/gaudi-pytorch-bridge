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
#include "generated/backend/_foreach_addcdiv.h"
#include "generated/backend/addcdiv.h"
#include "generated/backend/addcmul.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

OutputMetaData CompoundMetaCommon(
    const at::Tensor& self,
    const at::Tensor& other1,
    const at::Tensor& other2) {
  const at::ScalarType dtype =
      at::promote_types(self.scalar_type(), at::result_type(other1, other2));
  const std::vector<int64_t> shape = at::infer_size(
      at::infer_size(self.sizes(), other1.sizes()), other2.sizes());
  return {dtype, shape};
}

OutputMetaDataVector AddCOpsMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& other1 = stack_tensor(stack, 1);
  const torch::Tensor& other2 = stack_tensor(stack, 2);

  return {CompoundMetaCommon(self, other1, other2)};
}

static SharedMetaDataVector AddCompositeSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& self = stack_tensor(stack, 0);
  const auto& other1 = stack_tensor(stack, 1);
  const auto& other2 = stack_tensor(stack, 2);
  const bool tensor_value = stack.at(3).isTensor();
  const auto output_rank =
      std::max(std::max(self.dim(), other1.dim()), other2.dim());
  const at::ScalarType dtype =
      at::promote_types(self.scalar_type(), at::result_type(other1, other2));

  SharedMetaData meta{guid};
  meta.inputs_data = {
      {self.dim(), dtype}, {other1.dim(), dtype}, {other2.dim(), dtype}};
  if (tensor_value) {
    meta.inputs_data.push_back({0, dtype});
  }
  meta.outputs_data = {{output_rank, dtype}};

  return {meta};
}

SharedMetaDataVector AddCDivSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return AddCompositeSharedMeta(stack, "addcdiv_fwd");
}

SharedMetaDataVector AddCMulSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return AddCompositeSharedMeta(stack, "addcmul_fwd");
}

OutputMetaDataVector ForeachCompoundMeta(const at::Stack& stack) {
  const auto& selfs = stack.at(0).toTensorList();
  const auto& tensors1 = stack.at(1).toTensorList();
  const auto& tensors2 = stack.at(2).toTensorList();

  OutputMetaDataVector outputMetaDataVector;
  outputMetaDataVector.reserve(selfs.size());

  for (size_t i = 0; i < selfs.size(); ++i) {
    outputMetaDataVector.push_back(
        CompoundMetaCommon(selfs[i], tensors1[i], tensors2[i]));
  }

  return outputMetaDataVector;
}

SharedMetaDataVector ForeachAddcdivSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return ForeachCompoundSharedMeta(stack, "addcdiv_fwd");
}

SharedMetaDataVector ForeachAddcmulSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return ForeachCompoundSharedMeta(stack, "addcmul_fwd");
}

std::shared_ptr<void> FillAddCompositeParams(
    const at::Stack& stack,
    BinaryWithAlphaMode_t mode,
    size_t& size) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);

  params->mode = mode;
  // if alpha is not equal to 1 then it is passed as tensor (4th input),
  // otherwise as params
  auto scalar = stack.at(3).isScalar() ? stack.at(3).toScalar() : 1;
  auto meta = AddCOpsMeta(stack)[0];
  const bool isOutputIntegral = c10::isIntegralType(meta.dtype, true);
  const bool isAddcdiv =
      mode == BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CDIV;
  if (isOutputIntegral && !isAddcdiv) {
    params->alpha.i =
        scalar.isFloatingPoint() ? scalar.to<float>() : scalar.to<int>();
  } else {
    params->alpha.f =
        scalar.isFloatingPoint() ? scalar.to<float>() : scalar.to<int>();
  }

  return params;
}

std::shared_ptr<void> FillAddcmulParams(const at::Stack& stack, size_t& size) {
  return FillAddCompositeParams(
      stack, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CMUL, size);
}

std::shared_ptr<void> FillAddcdivParams(const at::Stack& stack, size_t& size) {
  return FillAddCompositeParams(
      stack, BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CDIV, size);
}

void ForeachCompound::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  HABANA_ASSERT(
      guid_.find("addcdiv") != std::string::npos ||
          guid_.find("addcmul") != std::string::npos,
      "Guid need to be addcdiv or addcmul");

  const auto& selfs = stack.at(0).toTensorList();
  const auto& value = stack.at(3);

  const bool isValueTensor = value.isTensor();
  bool isValueFloatingType = false;
  at::ScalarType valueType{};
  if (isValueTensor) {
    valueType = value.toTensor().scalar_type();
    isValueFloatingType = c10::isFloatingType(valueType);
  }

  const bool isAddcdiv = guid_.find("addcdiv") != std::string::npos;
  const size_t selfs_size = selfs.size();
  const auto metas = ForeachCompoundMeta(stack);

  std::optional<synapse_helpers::tensor> cast{};
  if (isValueTensor && valueType == torch::kInt64 &&
      (habana::HPUDeviceContext::get_device().type() !=
       synDeviceType::synDeviceGaudi)) {
    cast = BuildCast(
        this,
        graph,
        syn_in(3 * selfs_size),
        value.toTensor().sizes(),
        torch::kInt64,
        torch::kInt32);
  }

  ns_BinaryWithAlphaKernel::Params params{};
  params.mode = isAddcdiv ? BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CDIV
                          : BinaryWithAlphaMode_t::BINARY_WITH_ALPHA_MODE_CMUL;

  for (size_t i = 0; i < selfs_size; ++i) {
    std::vector<synTensor> inputs = {
        syn_in(i), syn_in(i + selfs_size), syn_in(i + 2 * selfs_size)};
    const bool isOutputIntegral = c10::isIntegralType(metas[i].dtype, true);
    guid_ = update_guid_dtype(
        guid_,
        isAddcdiv && isOutputIntegral ? torch::kFloat32 : metas[i].dtype);

    std::optional<synapse_helpers::tensor> floor{};
    std::vector<synapse_helpers::tensor> sliced{};

    if (isValueTensor) {
      synSliceAxisParamsV2 slice_params{};
      slice_params.axis = 0;
      slice_params.begin = i;
      slice_params.end = i + 1;
      sliced = BuildOp(
          graph,
          "slice_axis",
          {cast.has_value() ? cast.value().get() : syn_in(3 * selfs_size)},
          {{{1}, valueType == torch::kInt64 ? torch::kInt32 : valueType}},
          &slice_params,
          sizeof(slice_params));

      if (isValueFloatingType && isOutputIntegral) {
        floor = std::move(BuildOp(
            graph,
            get_guid_with_precision("floor_fwd", valueType),
            {sliced[0].get()},
            {{{1}, valueType}})[0]);
      }
      inputs.push_back(
          floor.has_value() ? floor.value().get() : sliced[0].get());
    } else {
      at::Scalar scalar =
          value.isScalar() ? value.toScalar() : value.toListRef()[i].toScalar();

      if (isOutputIntegral && !isAddcdiv) {
        params.alpha.i =
            scalar.isFloatingPoint() ? scalar.to<float>() : scalar.to<int>();
      } else {
        params.alpha.f =
            scalar.isFloatingPoint() ? scalar.to<float>() : scalar.to<int>();
      }
    }
    auto out = BuildOp(
        graph,
        guid_,
        std::move(inputs),
        {{metas[i].shape, metas[i].dtype, i}},
        &params,
        sizeof(params));
    syn_out(i) = std::move(out[0]);
  }
}

} // namespace habana
