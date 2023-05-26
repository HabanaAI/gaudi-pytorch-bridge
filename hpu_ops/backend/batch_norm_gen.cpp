/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <perf_lib_layer_params.h>
#include "backend/helpers/create_tensor.h"
#include "generated/backend/native_batch_norm.h"
#include "generated/backend/native_batch_norm_backward.h"

namespace habana {

namespace sh = synapse_helpers;

namespace BNFwd {

enum InputIdx {
  INPUT_IDX = 0,
  WEIGHT_IDX = 1,
  BIAS_IDX = 2,
  RUNNING_MEAN_IDX = 3,
  RUNNING_VAR_IDX = 4,
  IS_TRAINING_IDX = 5,
  MOMENTUM_IDX = 6,
  EPSILON_IDX = 7
};

enum OutputIdx { OUTPUT_IDX = 0, SAVED_MEAN_IDX = 1, SAVED_ISTD_IDX = 2 };

}; // namespace BNFwd

namespace BNBwd {

enum InputIdx {
  GRAD_OUT_IDX = 0,
  INPUT_IDX = 1,
  WEIGHT_IDX = 2,
  RUNNING_MEAN_IDX = 3,
  RUNNING_VAR_IDX = 4,
  SAVED_MEAN_IDX = 5,
  SAVED_ISTD_IDX = 6,
  IS_TRAINING_IDX = 7,
  EPSILON_IDX = 8
};

enum OutputIdx { INPUT_GRAD_IDX = 0, WEIGHT_GRAD_IDX = 1, BIAS_GRAD_IDX = 2 };

} // namespace BNBwd
inline bool is_training(bool pt_training_flag, bool running_mean_defined) {
  bool inference_mode = !pt_training_flag && running_mean_defined;
  if (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE)) {
    inference_mode = true;
  }
  return !inference_mode;
}

c10::IntArrayRef get_rm_size(const at::Tensor& input) {
  int rm_size_idx;
  switch (input.suggest_memory_format()) {
    case c10::MemoryFormat::ChannelsLast:
      rm_size_idx = 3;
      break;
    case c10::MemoryFormat::ChannelsLast3d:
      rm_size_idx = 4;
      break;
    default:
      rm_size_idx = 1;
      break;
  }
  return input.sizes()[rm_size_idx];
}

OutputMetaDataVector BatchNormFwdMeta(const at::Stack& stack) {
  using namespace BNFwd;
  const auto& input = stack[INPUT_IDX].toTensor();
  auto saved_mean_sv = stack[WEIGHT_IDX].isTensor()
      ? stack[WEIGHT_IDX].toTensor().sizes().vec()
      : get_rm_size(stack[INPUT_IDX].toTensor()).vec();
  auto saved_istd_sv = stack[BIAS_IDX].isTensor()
      ? stack[BIAS_IDX].toTensor().sizes().vec()
      : get_rm_size(stack[INPUT_IDX].toTensor()).vec();

  OutputMetaData out_meta;
  out_meta.shape = input.sizes().vec();
  out_meta.dtype = input.scalar_type();
  OutputMetaData saved_mean_meta;
  saved_mean_meta.shape = saved_mean_sv;
  saved_mean_meta.dtype = c10::ScalarType::Float;
  OutputMetaData saved_istd_meta;
  saved_istd_meta.shape = saved_istd_sv;
  saved_istd_meta.dtype = c10::ScalarType::Float;
  return {out_meta, saved_mean_meta, saved_istd_meta};
}

OutputMetaDataVector BatchNormFunctionalFwdMeta(const at::Stack& stack) {
  OutputMetaDataVector v = BatchNormFwdMeta(stack);

  OutputMetaData running_mean_meta = v[2];
  v.push_back(running_mean_meta);

  OutputMetaData running_var_meta = v[2];
  v.push_back(running_var_meta);

  return v;
}

std::shared_ptr<void> FillBatchNormFwdParams(
    const at::Stack& stack,
    size_t& size) {
  using namespace BNFwd;
  bool is_training_ = is_training(
      stack.at(IS_TRAINING_IDX).toBool(),
      stack.at(RUNNING_MEAN_IDX).isTensor());
  if (is_training_) {
    PARAMS_STUB(ns_BatchNormKernel::ParamsV2);
    params->momentum = static_cast<float>(stack.at(MOMENTUM_IDX).toDouble());
    params->epsilon = static_cast<float>(stack.at(EPSILON_IDX).toDouble());
    params->threshold.f = 0.0;
    params->isTraining = is_training_;
    return params;
  } else {
    PARAMS_STUB(ns_BatchNormKernel::Params);
    params->momentum = static_cast<float>(stack.at(MOMENTUM_IDX).toDouble());
    params->epsilon = static_cast<float>(stack.at(EPSILON_IDX).toDouble());
    params->threshold.f = 0.0;
    return params;
  }
}

std::shared_ptr<void> FillBatchNormBwdParams(
    const at::Stack& stack,
    size_t& size) {
  using namespace BNBwd;
  PARAMS_STUB(ns_BatchNormKernel::ParamsV2);
  params->momentum = 0.0;
  params->epsilon = static_cast<float>(stack.at(EPSILON_IDX).toDouble());
  params->threshold.f = 0.0;
  params->isTraining = stack.at(IS_TRAINING_IDX).toBool();
  return params;
}

sizes_vec BatchNormFwdOutputShape(const at::Stack& stack) {
  using namespace BNFwd;
  auto input_sv = stack[INPUT_IDX].toTensor().sizes().vec();
  auto mean_sv = stack[RUNNING_MEAN_IDX].isTensor()
      ? stack[RUNNING_MEAN_IDX].toTensor().sizes().vec()
      : get_rm_size(stack[INPUT_IDX].toTensor()).vec();
  auto var_sv = stack[RUNNING_VAR_IDX].isTensor()
      ? stack[RUNNING_VAR_IDX].toTensor().sizes().vec()
      : get_rm_size(stack[INPUT_IDX].toTensor()).vec();
  return {input_sv, mean_sv, var_sv};
}

sizes_vec BatchNormBwdOutputShape(const at::Stack& stack) {
  using namespace BNBwd;
  auto input_grad_sv = stack[INPUT_IDX].toTensor().sizes().vec();
  auto weight_grad_sv = stack[WEIGHT_IDX].toTensor().sizes().vec();
  auto bias_grad_sv = stack[WEIGHT_IDX].toTensor().sizes().vec();
  return {input_grad_sv, weight_grad_sv, bias_grad_sv};
}

sh::tensor get_4d_tensor(
    OpBackend* op,
    sh::graph& graph,
    const OpBackend::TensorsPair& input) {
  const auto in_shape = input.pt_t.sizes();

  std::vector<int64_t> ret_shape(4, 1);

  // TPC  BN supports only 4D inputs. This means that any higher dims have to
  // be flattened
  std::optional<sh::tensor> storage;
  if (in_shape.size() > 4) {
    // Input is in dims format N,C,D1,D2,D3,...,Dm,H,W
    std::vector<int64_t> permute_dims(in_shape.size(), 0);
    for (size_t i = 0; i < permute_dims.size(); i++) {
      permute_dims[i] = i;
    }
    std::swap(permute_dims[1], permute_dims[permute_dims.size() - 3]);
    // Changed/Permuted Input is in dims format N,Dm,D1,D2,D3,...,C,H,W
    storage = OpBackend::BuildPermute(
        op, graph, input.syn_t, in_shape, permute_dims, op->ScalarType());
    // in_ = in_t.permute(permute_dims);
    // in_shape = in_.sizes().vec();
    auto higher_dim_size = std::accumulate(
                               in_shape.begin() + 2,
                               in_shape.begin() + in_shape.size() - 2,
                               1,
                               std::multiplies<int64_t>{}) *
        in_shape[0];

    // Get the shape ready to change input to format
    // {(N*Dm*D1*D2*D3*Dm-1),C,H,W}
    ret_shape[0] = higher_dim_size;
    ret_shape[1] = in_shape[1];
    ret_shape[2] = in_shape[in_shape.size() - 2];
    ret_shape[3] = in_shape[in_shape.size() - 1];
  } else {
    std::copy(in_shape.begin(), in_shape.end(), ret_shape.begin());
  }
  // TODO how is this supposed to work?
  if (3 == in_shape.size()) { // For 3-D in_t[2] should be at reshaped_t[3]
    std::swap(ret_shape[2], ret_shape[3]);
  }

  auto output = OpBackend::BuildReshape(
      op,
      graph,
      storage ? (*storage).get() : input.syn_t,
      ret_shape,
      op->ScalarType());

  return output;
}

void get_orig_shape_tensor(
    OpBackend* op,
    sh::graph& graph,
    c10::IntArrayRef input_sizes,
    std::optional<sh::tensor>& postprocess_out_storage) {
  if (input_sizes.size() == 4) {
    return;
  }

  auto in_shape = input_sizes.vec();
  if (in_shape.size() >= 1 && in_shape.size() <= 3) {
    postprocess_out_storage = OpBackend::BuildReshape(
        op,
        graph,
        (*postprocess_out_storage).get(),
        in_shape,
        op->ScalarType(),
        0);
  } else {
    // Input is in dims format N,Dm,D1,D2,D3,...,C,H,W
    // Final output should be in format N,C,D1,D2,D3,...,Dm,H,W
    std::vector<int64_t> permute_dims(in_shape.size(), 0);
    for (size_t i = 0; i < permute_dims.size(); i++) {
      permute_dims[i] = i;
    }
    std::swap(permute_dims[1], permute_dims[permute_dims.size() - 3]);
    std::swap(in_shape[1], in_shape[in_shape.size() - 3]);

    postprocess_out_storage = OpBackend::BuildReshape(
        op,
        graph,
        (*postprocess_out_storage).get(),
        in_shape,
        op->ScalarType());
    postprocess_out_storage = OpBackend::BuildPermute(
        op,
        graph,
        (*postprocess_out_storage).get(),
        in_shape,
        permute_dims,
        op->ScalarType(),
        0);
  }
}

#define TRANSFORM_TO_4D(input)                                      \
  synTensor input##_4d = input.syn_t;                               \
  std::optional<sh::tensor> input##_4d_storage;                     \
  std::vector<int64_t> input##_4d_shape = input.pt_t.sizes().vec(); \
  if (input.pt_t.sizes().size() != 4) {                             \
    input##_4d_storage = get_4d_tensor(this, graph, input);         \
    input##_4d = (*input##_4d_storage).get();                       \
    input##_4d_shape = (*input##_4d_storage).pt_shape();            \
  }

#define GET_OR_CREATE_INPUT_STUB(in_name, fill_val)                        \
  synTensor in_name;                                                       \
  std::optional<sh::tensor> in_name##_storage;                             \
  std::vector<int64_t> in_name##_shape;                                    \
  std::variant<synapse_helpers::tensor*, int> in_name##_sh_t_or_idx;       \
  if (in_name##_opt.has_value()) {                                         \
    in_name = (*in_name##_opt).syn_t;                                      \
    in_name##_shape = in_name##_opt->pt_t.sizes().vec();                   \
    in_name##_sh_t_or_idx = (*in_name##_opt).syn_idx;                      \
  } else {                                                                 \
    in_name##_storage = ConstantHelper(                                    \
        graph, fill_val, c10::ScalarType::Float, get_rm_size(input.pt_t)); \
    in_name = (*in_name##_storage).get();                                  \
    in_name##_shape = (*in_name##_storage).pt_shape();                     \
    in_name##_sh_t_or_idx = in_name##_storage.operator->();                \
  }

void BatchNormOpBackend::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace BNFwd;

  /* 1. Collect inputs */
  StackGetter stackGetter(stack, "BatchNormOpBackend::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto weight_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto bias_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto running_mean_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto running_var_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool training = getNextInput<bool>(stackGetter);

  bool is_functional = GetGuid().find("_native_batch_norm_legit_functional") !=
      std::string::npos;

  /* 2. Perform frontend operations */
  // In case of batch norm:
  // 2.1 Preprocess inputs
  // 2.1.1 Reshape input to 4D
  TRANSFORM_TO_4D(input);

  /* 2.1.2 Create weight, bias, running_mean, running_var if not defined */
  auto rm_size = get_rm_size(input.pt_t);

  GET_OR_CREATE_INPUT_STUB(weight, 1);
  GET_OR_CREATE_INPUT_STUB(bias, 0);
  GET_OR_CREATE_INPUT_STUB(running_mean, 0);
  GET_OR_CREATE_INPUT_STUB(running_var, 1);

  auto out_shapes = BatchNormFwdOutputShape(stack);

  size_t size; // Will be initialized by below call
  const auto& params = FillBatchNormFwdParams(stack, size);

  std::vector<sh::tensor> bn_out;
  bn_out.reserve(3);

  c10::optional<int> final_result_index_0 = input.pt_t.sizes().size() != 4
      ? c10::optional<int>{c10::nullopt}
      : c10::optional<int>{0};
  if (is_training(training, running_mean_opt.has_value())) {
    // 2.2 Handle training
    // 2.2.1 Call native_batch_norm_training
    // 2.2.2 Handle RUNNING_HASH... macros
    bn_out = BuildOp(
        graph,
        get_guid_with_precision("batch_norm_fwd", ScalarType()),
        {input_4d, bias, weight, running_mean, running_var},
        {NodeAttr::NodeOutputAttr{
             input_4d_shape, ScalarType(), final_result_index_0},
         NodeAttr::NodeOutputAttr{
             out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 1},
         NodeAttr::NodeOutputAttr{
             out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, 2},
         is_functional
             ? NodeAttr::
                   NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 3}
             : NodeAttr::
                   NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, c10::nullopt, DATA_TENSOR, syn_type_na, running_mean_sh_t_or_idx},
         is_functional
             ? NodeAttr::
                   NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 4}
             : NodeAttr::
                   NodeOutputAttr{out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, c10::nullopt, DATA_TENSOR, syn_type_na, running_var_sh_t_or_idx}},
        params.get(),
        size);

    if (running_mean_opt.has_value() && !is_functional) {
      GetSynImplicitOutputs().emplace_back(PtInputIdxAndSynHelpTensor{
          3, std::move(bn_out[3]), std::get<int>(running_mean_sh_t_or_idx)});
    }
    if (running_var_opt.has_value() && !is_functional) {
      GetSynImplicitOutputs().emplace_back(PtInputIdxAndSynHelpTensor{
          4, std::move(bn_out[4]), std::get<int>(running_var_sh_t_or_idx)});
    }
  } else {
    // 2.3 Handle inference
    // 2.3.1 Call native_batch_norm_inf
    bn_out.emplace_back(
        std::move(BuildOp(
                      graph,
                      get_guid_with_precision("batch_norm_inf", ScalarType()),
                      {input_4d, bias, weight, running_mean, running_var},
                      {{input_4d_shape, ScalarType(), final_result_index_0}},
                      params.get(),
                      size)
                      .at(0)));

    bn_out.emplace_back(
        std::move(BuildOp(
                      graph,
                      "identity",
                      {running_mean},
                      {{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 1}})
                      .at(0)));

    bn_out.emplace_back(
        std::move(BuildOp(
                      graph,
                      "identity",
                      {running_var},
                      {{out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, 2}})
                      .at(0)));

    if (is_functional) {
      bn_out.emplace_back(std::move(
          BuildOp(
              graph,
              "identity",
              {running_mean},
              {{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 3}})
              .at(0)));

      bn_out.emplace_back(std::move(
          BuildOp(
              graph,
              "identity",
              {running_var},
              {{out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, 4}})
              .at(0)));
    }
  }

  // 2.4 Postprocess outputs
  // 2.4.1 Reshape output to original input's shape
  sh::tensor* postprocess_out = &bn_out[0];
  std::optional<sh::tensor> postprocess_out_storage =
      std::move(*postprocess_out);

  get_orig_shape_tensor(
      this, graph, input.pt_t.sizes(), postprocess_out_storage);

  postprocess_out = &(*postprocess_out_storage);

  syn_out(0) = std::move(*postprocess_out);
  syn_out(1) = std::move(bn_out[1]);
  syn_out(2) = std::move(bn_out[2]);
  if (is_functional) {
    syn_out(3) = std::move(bn_out[3]);
    syn_out(4) = std::move(bn_out[4]);
  }
}

void BatchNormBwdOpBackend::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace BNBwd;

  /* 1. Collect inputs */
  StackGetter stackGetter(stack, "BatchNormFwdOpBackend::AddNode");
  auto grad_out = getNextInput<TensorsPair>(stackGetter);
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto weight_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto running_mean_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto running_var_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto saved_mean_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto saved_istd_opt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool training = getNextInput<bool>(stackGetter);
  double eps = getNextInput<double>(stackGetter);

  /* 2. Perform frontend operations */
  // In case of batch norm:
  // 2.1 Preprocess inputs
  // 2.1.1 Reshape input to 4D
  TRANSFORM_TO_4D(input);
  TRANSFORM_TO_4D(grad_out);

  GET_OR_CREATE_INPUT_STUB(weight, 1);

  synTensor saved_mean, saved_istd;
  std::optional<sh::tensor> saved_mean_storage, saved_istd_storage;
  if (!is_training(training, running_mean_opt.has_value())) {
    GET_OR_CREATE_INPUT_STUB(running_mean, 0);
    GET_OR_CREATE_INPUT_STUB(running_var, 1);

    // TODO calculate saved_mean, saved_istd
    saved_mean = running_mean;

    // istd = rsqrt(add(running_var, eps));
    auto eps_constant = ConstantHelper(graph, eps, c10::ScalarType::Float, {1});

    auto rv_add_eps =
        std::move(BuildOp(
                      graph,
                      "add_fwd_f32",
                      {running_var, eps_constant.get()},
                      {{running_var_shape, c10::ScalarType::Float}})
                      .at(0));

    saved_istd_storage =
        std::move(BuildOp(
                      graph,
                      "rsqrt_fwd_f32",
                      {rv_add_eps.get()},
                      {{running_var_shape, c10::ScalarType::Float}})
                      .at(0));
    saved_istd = (*saved_istd_storage).get();
  } else {
    saved_mean = (*saved_mean_opt).syn_t;
    saved_istd = (*saved_istd_opt).syn_t;
  }

  auto out_shapes = BatchNormBwdOutputShape(stack);

  size_t size; // Will be initialized by below call
  const auto& params = FillBatchNormBwdParams(stack, size);

  c10::optional<int> final_result_index_0 = input.pt_t.sizes().size() != 4
      ? c10::optional<int>{c10::nullopt}
      : c10::optional<int>{INPUT_GRAD_IDX};
  auto bn_out = BuildOp(
      graph,
      get_guid_with_precision("batch_norm_bwd", ScalarType()),
      {input_4d, grad_out_4d, saved_mean, saved_istd, weight},
      {{input_4d_shape, ScalarType(), final_result_index_0},
       {out_shapes[BIAS_GRAD_IDX], c10::ScalarType::Float, BIAS_GRAD_IDX},
       {out_shapes[WEIGHT_GRAD_IDX], c10::ScalarType::Float, WEIGHT_GRAD_IDX}},
      params.get(),
      size);

  // 2.4 Postprocess outputs
  // 2.4.1 Reshape output to original input's shape
  sh::tensor* postprocess_out = &bn_out[0];
  std::optional<sh::tensor> postprocess_out_storage =
      std::move(*postprocess_out);

  get_orig_shape_tensor(
      this, graph, input.pt_t.sizes(), postprocess_out_storage);

  postprocess_out = &(*postprocess_out_storage);

  syn_out(INPUT_GRAD_IDX) = std::move(*postprocess_out);
  syn_out(WEIGHT_GRAD_IDX) = std::move(bn_out[2]);
  syn_out(BIAS_GRAD_IDX) = std::move(bn_out[1]);
}

} // namespace habana
