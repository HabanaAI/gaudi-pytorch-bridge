/*******************************************************************************
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
#include "backend/helpers/runtime_config.h"
#include "generated/backend/_native_batch_norm_legit.h"
#include "generated/backend/_native_batch_norm_legit_no_training.h"
#include "generated/backend/native_batch_norm.h"
#include "generated/backend/native_batch_norm_backward.h"

namespace habana {

namespace sh = synapse_helpers;

namespace {
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

namespace BNNoTrainingFwd {

enum InputIdx {
  INPUT_IDX = 0,
  WEIGHT_IDX = 1,
  BIAS_IDX = 2,
  RUNNING_MEAN_IDX = 3,
  RUNNING_VAR_IDX = 4,
  MOMENTUM_IDX = 5,
  EPSILON_IDX = 6
};

};

namespace BNNoStatsFwd {

enum InputIdx {
  INPUT_IDX = 0,
  WEIGHT_IDX = 1,
  BIAS_IDX = 2,
  IS_TRAINING_IDX = 3,
  MOMENTUM_IDX = 4,
  EPSILON_IDX = 5
};

};

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

inline bool is_training(bool pt_training_flag, bool is_running_mean_defined) {
  bool inference_mode = (not pt_training_flag) and is_running_mean_defined;
  return habana_helpers::IsInferenceMode() ? false : not inference_mode;
}

template <typename T>
std::tuple<std::shared_ptr<void>, size_t> fillBatchNormParams(
    float momentum,
    float epsilon) {
  size_t size;
  PARAMS_STUB(T);
  params->momentum = momentum;
  params->epsilon = epsilon;
  params->threshold.f = 0.0;
  if constexpr (std::is_same_v<T, ns_BatchNormKernel::ParamsV2>) {
    params->isTraining = true;
  }

  return std::make_tuple(params, size);
}

auto fillBatchNormParams(bool isTraining, float momentum, float epsilon) {
  return (isTraining)
      ? fillBatchNormParams<ns_BatchNormKernel::ParamsV2>(momentum, epsilon)
      : fillBatchNormParams<ns_BatchNormKernel::Params>(momentum, epsilon);
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

sh::tensor get_4d_tensor(
    OpBackend& op,
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
        &op, graph, input.syn_t, in_shape, permute_dims, op.ScalarType());
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

  return OpBackend::BuildReshape(
      &op,
      graph,
      storage ? (*storage).get() : input.syn_t,
      ret_shape,
      op.ScalarType());
}

void reshape_tensor(
    OpBackend& op,
    sh::graph& graph,
    c10::IntArrayRef input_sizes,
    sh::tensor& inout_tensor) {
  if (input_sizes.size() == 4) {
    return;
  }

  auto in_shape = input_sizes.vec();
  if (in_shape.size() >= 1 && in_shape.size() <= 3) {
    inout_tensor = OpBackend::BuildReshape(
        &op, graph, inout_tensor.get(), in_shape, op.ScalarType(), 0);
  } else {
    // Input is in dims format N,Dm,D1,D2,D3,...,C,H,W
    // Final output should be in format N,C,D1,D2,D3,...,Dm,H,W
    std::vector<int64_t> permute_dims(in_shape.size(), 0);
    for (size_t i = 0; i < permute_dims.size(); i++) {
      permute_dims[i] = i;
    }
    std::swap(permute_dims[1], permute_dims[permute_dims.size() - 3]);
    std::swap(in_shape[1], in_shape[in_shape.size() - 3]);

    inout_tensor = OpBackend::BuildReshape(
        &op, graph, inout_tensor.get(), in_shape, op.ScalarType());
    inout_tensor = OpBackend::BuildPermute(
        &op,
        graph,
        inout_tensor.get(),
        in_shape,
        permute_dims,
        op.ScalarType(),
        0);
  }
}

std::tuple<synTensor, std::vector<int64_t>, std::optional<sh::tensor>>
transform_tensor_to_4d(
    OpBackend& op,
    sh::graph& graph,
    const OpBackend::TensorsPair& tensor) {
  if (tensor.pt_t.sizes().size() != 4) {
    sh::tensor tensor_4d_storage = get_4d_tensor(op, graph, tensor);
    synTensor tensor_4d_syn = tensor_4d_storage.get();
    std::vector<int64_t> tensor_4d_shape = tensor_4d_storage.pt_shape();
    return std::make_tuple(
        std::move(tensor_4d_syn),
        std::move(tensor_4d_shape),
        std::move(tensor_4d_storage));
  }
  return std::make_tuple(tensor.syn_t, tensor.pt_t.sizes().vec(), std::nullopt);
}

std::tuple<synTensor, std::vector<int64_t>, std::variant<sh::tensor, int>>
get_or_create_tensor(
    OpBackend& op,
    sh::graph& graph,
    const c10::optional<OpBackend::TensorsPair>& tensor,
    const c10::IntArrayRef& rm_size,
    const at::Scalar& val) {
  if (not tensor.has_value()) {
    sh::tensor stub_tensor_storage =
        op.BuildConstant(&op, graph, val, c10::ScalarType::Float, rm_size);
    auto stub_tensor_syn = stub_tensor_storage.get();
    auto stub_tensor_shape = stub_tensor_storage.pt_shape();
    return std::make_tuple(
        std::move(stub_tensor_syn),
        std::move(stub_tensor_shape),
        std::move(stub_tensor_storage));
  }
  return std::make_tuple(
      (*tensor).syn_t, (*tensor).pt_t.sizes().vec(), (*tensor).syn_idx);
}

std::variant<sh::tensor*, int> variant_cast(std::variant<sh::tensor, int>& v) {
  if (std::holds_alternative<sh::tensor>(v)) {
    return &std::get<sh::tensor>(v);
  }
  return std::get<int>(v);
}

auto get_running_var_def_value(const OpBackend& op) {
  bool is_no_stats = op.GetGuid().find("_native_batch_norm_legit.no_stats") !=
      std::string::npos;

  return is_no_stats ? 0 : 1;
}

bool is_batch_norm_functional(const OpBackend& op) {
  return op.GetGuid().find("_native_batch_norm_legit_functional") !=
      std::string::npos;
}

namespace {
template <typename... Args>
inline void unused_variables(const Args&...){};
}

std::vector<sh::tensor> handle_batch_norm_training_fwd(
    OpBackend& op,
    sh::graph& graph,
    const OpBackend::TensorsPair& input,
    const c10::optional<OpBackend::TensorsPair>& weight_opt,
    const c10::optional<OpBackend::TensorsPair>& bias_opt,
    const c10::optional<OpBackend::TensorsPair>& running_mean_opt,
    const c10::optional<OpBackend::TensorsPair>& running_var_opt,
    const std::shared_ptr<void>& params,
    const size_t params_size,
    const sizes_vec& out_shapes) {
  using namespace BNFwd;

  const auto [input_4d, input_4d_shape, input_4d_storage] =
      transform_tensor_to_4d(op, graph, input);

  c10::IntArrayRef rm_size = get_rm_size(input.pt_t);
  const auto [weight, weight_shape, weight_storage_or_idx] =
      get_or_create_tensor(op, graph, weight_opt, rm_size, 1);
  const auto [bias, bias_shape, bias_storage_or_idx] =
      get_or_create_tensor(op, graph, bias_opt, rm_size, 0);
  auto [running_mean, running_mean_shape, running_mean_storage_or_idx] =
      get_or_create_tensor(op, graph, running_mean_opt, rm_size, 0);
  auto [running_var, running_var_shape, running_var_storage_or_idx] =
      get_or_create_tensor(
          op, graph, running_var_opt, rm_size, get_running_var_def_value(op));

  unused_variables(
      input_4d_storage,
      weight_shape,
      weight_storage_or_idx,
      bias_shape,
      bias_storage_or_idx,
      running_mean_shape,
      running_var_shape);

  auto input_dim = input.pt_t.sizes().size();
  bool is_functional = is_batch_norm_functional(op);

  std::vector<sh::tensor> bn_out = OpBackend::BuildNode(
      &op,
      graph,
      {get_guid_with_precision("batch_norm_fwd", op.ScalarType()),
       {input_4d, bias, weight, running_mean, running_var},
       {NodeAttr::NodeOutputAttr{
            input_4d_shape,
            op.ScalarType(),
            (input_dim != 4) ? c10::nullopt : c10::optional<int>(0)},
        NodeAttr::NodeOutputAttr{
            out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 1},
        NodeAttr::NodeOutputAttr{
            out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, 2},
        is_functional
            ? NodeAttr::
                  NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 3}
            : NodeAttr::
                  NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, c10::nullopt, DATA_TENSOR, syn_type_na, variant_cast(running_mean_storage_or_idx)}, // SAVED_ISTD_IDX?!
        is_functional
            ? NodeAttr::
                  NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 4}
            : NodeAttr::
                  NodeOutputAttr{out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, c10::nullopt, DATA_TENSOR, syn_type_na, variant_cast(running_var_storage_or_idx)}},
       params.get(),
       params_size});

  if (running_mean_opt.has_value() && not is_functional) {
    op.GetSynImplicitOutputs().emplace_back(PtInputIdxAndSynHelpTensor{
        3, std::move(bn_out[3]), std::get<int>(running_mean_storage_or_idx)});
  }
  if (running_var_opt.has_value() && not is_functional) {
    op.GetSynImplicitOutputs().emplace_back(PtInputIdxAndSynHelpTensor{
        4, std::move(bn_out[4]), std::get<int>(running_var_storage_or_idx)});
  }

  return bn_out;
}

std::vector<sh::tensor> handle_batch_norm_inference_fwd(
    OpBackend& op,
    sh::graph& graph,
    const OpBackend::TensorsPair& input,
    const c10::optional<OpBackend::TensorsPair>& weight_opt,
    const c10::optional<OpBackend::TensorsPair>& bias_opt,
    const c10::optional<OpBackend::TensorsPair>& running_mean_opt,
    const c10::optional<OpBackend::TensorsPair>& running_var_opt,
    const std::shared_ptr<void>& params,
    const size_t params_size,
    const sizes_vec& out_shapes) {
  using namespace BNFwd;

  std::vector<sh::tensor> bn_out;
  bn_out.reserve(5);

  const auto [input_4d, input_4d_shape, input_4d_storage] =
      transform_tensor_to_4d(op, graph, input);

  c10::IntArrayRef rm_size = get_rm_size(input.pt_t);
  const auto [weight, weight_size, weight_storage_or_idx] =
      get_or_create_tensor(op, graph, weight_opt, rm_size, 1);
  const auto [bias, bias_shape, bias_storage_or_idx] =
      get_or_create_tensor(op, graph, bias_opt, rm_size, 0);
  const auto [running_mean, running_mean_shape, running_mean_storage_or_idx] =
      get_or_create_tensor(op, graph, running_mean_opt, rm_size, 0);
  const auto [running_var, running_var_shape, running_var_storage_or_idx] =
      get_or_create_tensor(
          op, graph, running_var_opt, rm_size, get_running_var_def_value(op));

  unused_variables(
      input_4d_storage,
      weight_size,
      weight_storage_or_idx,
      bias_shape,
      bias_storage_or_idx,
      running_mean_shape,
      running_mean_storage_or_idx,
      running_var_shape,
      running_var_storage_or_idx);

  auto input_dim = input.pt_t.sizes().size();

  bn_out.emplace_back(std::move(
      OpBackend::BuildNode(
          &op,
          graph,
          {get_guid_with_precision("batch_norm_inf", op.ScalarType()),
           {input_4d, bias, weight, running_mean, running_var},
           {{input_4d_shape,
             op.ScalarType(),
             (input_dim != 4) ? c10::nullopt : c10::optional<int>(0)}},
           params.get(),
           params_size})
          .at(0)));

  bn_out.emplace_back(
      std::move(OpBackend::BuildNode(
                    &op,
                    graph,
                    {"identity",
                     {running_mean},
                     {{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 1}}})
                    .at(0)));

  bn_out.emplace_back(
      std::move(OpBackend::BuildNode(
                    &op,
                    graph,
                    {"identity",
                     {running_var},
                     {{out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, 2}}})
                    .at(0)));

  if (is_batch_norm_functional(op)) {
    bn_out.emplace_back(std::move(
        OpBackend::BuildNode(
            &op,
            graph,
            {"identity",
             {running_mean},
             {{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 3}}})
            .at(0)));

    bn_out.emplace_back(std::move(
        OpBackend::BuildNode(
            &op,
            graph,
            {"identity",
             {running_var},
             {{out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, 4}}})
            .at(0)));
  }

  return bn_out;
}
} // namespace

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

sizes_vec BatchNormNoStatsFwdOutputShape(const at::Stack& stack) {
  using namespace BNNoStatsFwd;
  auto input_sv = stack[INPUT_IDX].toTensor().sizes().vec();
  auto mean_sv = get_rm_size(stack[INPUT_IDX].toTensor()).vec();
  auto var_sv = get_rm_size(stack[INPUT_IDX].toTensor()).vec();
  return {input_sv, mean_sv, var_sv};
}

sizes_vec BatchNormBwdOutputShape(const at::Stack& stack) {
  using namespace BNBwd;
  auto input_grad_sv = stack[INPUT_IDX].toTensor().sizes().vec();
  auto weight_grad_sv = stack[WEIGHT_IDX].isTensor()
      ? stack[WEIGHT_IDX].toTensor().sizes().vec()
      : get_rm_size(stack[INPUT_IDX].toTensor()).vec();
  auto bias_grad_sv = stack[WEIGHT_IDX].isTensor()
      ? stack[WEIGHT_IDX].toTensor().sizes().vec()
      : get_rm_size(stack[INPUT_IDX].toTensor()).vec();
  return {input_grad_sv, weight_grad_sv, bias_grad_sv};
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
  float momentum = static_cast<float>(stack.at(MOMENTUM_IDX).toDouble());
  float epsilon = static_cast<float>(stack.at(EPSILON_IDX).toDouble());
  bool is_training_ = is_training(
      stack.at(IS_TRAINING_IDX).toBool(),
      stack.at(RUNNING_MEAN_IDX).isTensor());
  auto [params, paramsSize] =
      fillBatchNormParams(is_training_, momentum, epsilon);

  size = paramsSize;
  return params;
}

std::shared_ptr<void> FillBatchNormNoTrainingFwdParams(
    const at::Stack& stack,
    size_t& size) {
  using namespace BNNoTrainingFwd;
  float momentum = static_cast<float>(stack.at(MOMENTUM_IDX).toDouble());
  float epsilon = static_cast<float>(stack.at(EPSILON_IDX).toDouble());
  bool is_training_ = is_training(false, stack.at(RUNNING_MEAN_IDX).isTensor());
  auto [params, paramsSize] =
      fillBatchNormParams(is_training_, momentum, epsilon);

  size = paramsSize;
  return params;
}

std::shared_ptr<void> FillBatchNormNoStatsFwdParams(
    const at::Stack& stack,
    size_t& size) {
  using namespace BNNoStatsFwd;
  float momentum = static_cast<float>(stack.at(MOMENTUM_IDX).toDouble());
  float epsilon = static_cast<float>(stack.at(EPSILON_IDX).toDouble());
  bool is_training_ = is_training(stack.at(IS_TRAINING_IDX).toBool(), false);
  auto [params, paramsSize] =
      fillBatchNormParams(is_training_, momentum, epsilon);

  size = paramsSize;
  return params;
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

void BatchNormOpBackend::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "BatchNormOpBackend::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto weightOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto biasOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto runningMeanOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto runningVarOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool training = getNextInput<bool>(stackGetter);
  getNextInput<double>(stackGetter); // momentum
  getNextInput<double>(stackGetter); // epsilon

  size_t paramsSize; // Will be initialized by below call
  const auto params = FillBatchNormFwdParams(stack, paramsSize);
  const auto outShapes = BatchNormFwdOutputShape(stack);

  std::vector<sh::tensor> bnOut =
      (is_training(training, runningMeanOpt.has_value())
           ? handle_batch_norm_training_fwd
           : handle_batch_norm_inference_fwd)(
          *this,
          graph,
          input,
          weightOpt,
          biasOpt,
          runningMeanOpt,
          runningVarOpt,
          params,
          paramsSize,
          outShapes);

  reshape_tensor(*this, graph, input.pt_t.sizes(), bnOut[0]);

  syn_out(0) = std::move(bnOut[0]);
  syn_out(1) = std::move(bnOut[1]);
  syn_out(2) = std::move(bnOut[2]);
  if (is_batch_norm_functional(*this)) {
    syn_out(3) = std::move(bnOut[3]);
    syn_out(4) = std::move(bnOut[4]);
  }
}

void BatchNormNoTrainingOpBackend::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "BatchNormNoTrainingOpBackend::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto weightOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto biasOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto runningMeanOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto runningVarOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  getNextInput<double>(stackGetter); // momentum
  getNextInput<double>(stackGetter); // epsilon

  size_t paramsSize; // Will be initialized by below call
  const auto params = FillBatchNormNoTrainingFwdParams(stack, paramsSize);
  const auto outShapes = BatchNormFwdOutputShape(stack);

  std::vector<sh::tensor> bnOut =
      (is_training(false, runningMeanOpt.has_value())
           ? handle_batch_norm_training_fwd
           : handle_batch_norm_inference_fwd)(
          *this,
          graph,
          stack,
          input,
          weightOpt,
          biasOpt,
          runningMeanOpt,
          runningVarOpt,
          params,
          paramsSize,
          outShapes);

  reshape_tensor(*this, graph, input.pt_t.sizes(), bnOut[0]);

  syn_out(0) = std::move(bnOut[0]);
  syn_out(1) = std::move(bnOut[1]);
  syn_out(2) = std::move(bnOut[2]);
  if (is_batch_norm_functional(*this)) {
    syn_out(3) = std::move(bnOut[3]);
    syn_out(4) = std::move(bnOut[4]);
  }
}

void BatchNormNoStatsOpBackend::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "BatchNormNoStatsOpBackend::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto weightOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto biasOpt = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  bool training = getNextInput<bool>(stackGetter);
  getNextInput<double>(stackGetter); // momentum
  getNextInput<double>(stackGetter); // epsilon

  size_t paramsSize; // Will be initialized by below call
  const auto params = FillBatchNormNoStatsFwdParams(stack, paramsSize);
  const auto outShapes = BatchNormNoStatsFwdOutputShape(stack);

  std::vector<sh::tensor> bnOut =
      (is_training(training, false) ? handle_batch_norm_training_fwd
                                    : handle_batch_norm_inference_fwd)(
          *this,
          graph,
          input,
          weightOpt,
          biasOpt,
          c10::nullopt,
          c10::nullopt,
          params,
          paramsSize,
          outShapes);

  reshape_tensor(*this, graph, input.pt_t.sizes(), bnOut[0]);

  syn_out(0) = std::move(bnOut[0]);
  syn_out(1) = std::move(bnOut[1]);
  syn_out(2) = std::move(bnOut[2]);
  if (is_batch_norm_functional(*this)) {
    syn_out(3) = std::move(bnOut[3]);
    syn_out(4) = std::move(bnOut[4]);
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

  const auto [input_4d, input_4d_shape, input_4d_storage] =
      transform_tensor_to_4d(*this, graph, input);
  const auto [grad_out_4d, grad_out_4d_shape, grad_out_4d_storage] =
      transform_tensor_to_4d(*this, graph, grad_out);

  c10::IntArrayRef rm_size = get_rm_size(input.pt_t);
  const auto [weight, weight_shape, weight_storage_or_idx] =
      get_or_create_tensor(*this, graph, weight_opt, rm_size, 1);

  unused_variables(
      input_4d_storage,
      grad_out_4d_shape,
      grad_out_4d_storage,
      weight_shape,
      weight_storage_or_idx);

  synTensor saved_mean, saved_istd;
  std::optional<sh::tensor> saved_mean_storage, saved_istd_storage;
  if (!is_training(training, running_mean_opt.has_value())) {
    const auto [running_mean, running_mean_shape, running_mean_storage_or_idx] =
        get_or_create_tensor(*this, graph, running_mean_opt, rm_size, 0);
    const auto [running_var, running_var_shape, running_var_storage_or_idx] =
        get_or_create_tensor(*this, graph, running_var_opt, rm_size, 1);

    unused_variables(
        running_mean_shape,
        running_mean_storage_or_idx,
        running_var_storage_or_idx);

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
  const auto params = FillBatchNormBwdParams(stack, size);

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
  reshape_tensor(*this, graph, input.pt_t.sizes(), bn_out[0]);

  syn_out(INPUT_GRAD_IDX) = std::move(bn_out[0]);
  syn_out(WEIGHT_GRAD_IDX) = std::move(bn_out[2]);
  syn_out(BIAS_GRAD_IDX) = std::move(bn_out[1]);
}

} // namespace habana
