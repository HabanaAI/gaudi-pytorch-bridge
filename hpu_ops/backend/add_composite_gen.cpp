/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/_foreach_addcdiv.h"
#include "generated/backend/addcdiv.h"
#include "generated/backend/addcmul.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {

OutputMetaDataVector AddCOpsMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  const torch::Tensor& other1 = stack_tensor(stack, 1);
  const torch::Tensor& other2 = stack_tensor(stack, 2);
  auto tmp = at::infer_size(self.sizes(), other1.sizes());
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = at::infer_size(tmp, other2.sizes());
  return {meta};
}

static SharedMetaDataVector AddCompositeSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& self = stack_tensor(stack, 0);
  const auto self_dtype = self.scalar_type();
  const auto& other1 = stack_tensor(stack, 1);
  const auto& other2 = stack_tensor(stack, 2);
  const bool tensor_value = stack.at(3).isTensor();
  const auto output_rank =
      std::max(std::max(self.dim(), other1.dim()), other2.dim());

  SharedMetaData meta{guid};
  meta.inputs_data = {
      {self.dim(), self_dtype},
      {other1.dim(), other1.scalar_type()},
      {other2.dim(), other2.scalar_type()}};
  if (tensor_value) {
    meta.inputs_data.push_back({0, self_dtype});
  }
  meta.outputs_data = {{output_rank, self_dtype}};

  return {meta};
}

SharedMetaDataVector AddCDivSharedMeta(const at::Stack& stack) {
  return AddCompositeSharedMeta(stack, "addcdiv_fwd");
}

SharedMetaDataVector AddCMulSharedMeta(const at::Stack& stack) {
  return AddCompositeSharedMeta(stack, "addcmul_fwd");
}

OutputMetaDataVector ForeachCompoundMeta(const at::Stack& stack) {
  const auto& selfs = stack.at(0).toTensorList();
  const auto& tensors1 = stack.at(1).toTensorList();
  const auto& tensors2 = stack.at(2).toTensorList();

  OutputMetaDataVector outputMetaDataVector;
  outputMetaDataVector.reserve(selfs.size());

  for (size_t i = 0; i < selfs.size(); ++i) {
    const at::ScalarType dtype = at::promote_types(
        selfs[i].scalar_type(), at::result_type(tensors1[i], tensors2[i]));
    const std::vector<int64_t> shape = at::infer_size(
        at::infer_size(selfs[i].sizes(), tensors1[i].sizes()),
        tensors2[i].sizes());
    outputMetaDataVector.emplace_back(dtype, shape);
  }

  return outputMetaDataVector;
}

SharedMetaDataVector ForeachAddcdivSharedMeta(const at::Stack& stack) {
  return ForeachCompoundSharedMeta(stack, "addcdiv_fwd");
}

SharedMetaDataVector ForeachAddcmulSharedMeta(const at::Stack& stack) {
  return ForeachCompoundSharedMeta(stack, "addcmul_fwd");
}

std::shared_ptr<void> FillAddCompositeParams(
    const at::Stack& stack,
    BinaryWithAlphaMode_t mode,
    size_t& size) {
  PARAMS_STUB(ns_BinaryWithAlphaKernel::Params);
  auto out_scalar_type = stack.at(0).toTensor().scalar_type();

  params->mode = mode;
  // if alpha is not equal to 1 then it is passed as tensor (4th input),
  // otherwise as params
  auto val = stack.at(3).isScalar() ? stack.at(3).toScalar() : 1;
  if (c10::isFloatingType(out_scalar_type)) {
    get<float>(params->alpha) = val.to<float>();
  } else {
    get<int>(params->alpha) = val.to<int>();
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
  const auto& tensors1 = stack.at(1).toTensorList();
  const auto& tensors2 = stack.at(2).toTensorList();
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
      (habana::HPURegistrar::get_device().type() !=
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
