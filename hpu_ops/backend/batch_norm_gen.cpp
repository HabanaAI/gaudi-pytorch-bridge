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

#include <perf_lib_layer_params.h>
#include "backend/helpers/runtime_config.h"
#include "generated/backend/_native_batch_norm_legit.h"
#include "generated/backend/_native_batch_norm_legit_no_training.h"
#include "generated/backend/native_batch_norm.h"
#include "generated/backend/native_batch_norm_backward.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

namespace sh = synapse_helpers;

static bool should_cast_from_BF16(c10::optional<TensorsPair> tensor_pair_opt) {
  if (tensor_pair_opt.has_value())
    return tensor_pair_opt->pt_t.scalar_type() == c10::ScalarType::BFloat16;
  return false;
}

static synTensor cast_if_necessary_or_default(
    OpBackend* op,
    sh::graph& graph,
    c10::optional<TensorsPair> source_opt,
    synTensor& default_val,
    std::optional<sh::tensor>& storage) {
  if (should_cast_from_BF16(source_opt)) {
    storage = OpBackend::BuildCast(
        op,
        graph,
        source_opt->syn_t,
        source_opt->pt_t.sizes().vec(),
        c10::ScalarType::BFloat16,
        c10::ScalarType::Float);
    return storage->get();
  }
  return default_val;
}

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

synapse_helpers::layouts::SynapseLayoutFormat getSynapseLayout(const int64_t& dimensions)
{
    switch (dimensions) {
    case 2:
      return synapse_helpers::layouts::SynapseLayoutFormat::CN;
      break;
    case 3:
      return synapse_helpers::layouts::SynapseLayoutFormat::LCN;
      break;
    case 4:
      return synapse_helpers::layouts::SynapseLayoutFormat::WHCN;
      break;
    case 5:
      return synapse_helpers::layouts::SynapseLayoutFormat::WHDCN;
      break;
    default:
      return synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE;
      break;
  }
}

sh::tensor get_4d_tensor(
    OpBackend& op,
    sh::graph& graph,
    const TensorsPair& input) {
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
    sh::tensor& inout_tensor,
    c10::ScalarType scalarType) {
  if (input_sizes.size() == 4)
    return;

  auto in_shape = input_sizes.vec();
  if (in_shape.size() >= 1 && in_shape.size() <= 3) {
    inout_tensor = OpBackend::BuildReshape(
        &op, graph, inout_tensor.get(), in_shape, scalarType, 0);
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
        &op, graph, inout_tensor.get(), in_shape, scalarType);
    inout_tensor = OpBackend::BuildPermute(
        &op, graph, inout_tensor.get(), in_shape, permute_dims, scalarType, 0);
  }
}

template <unsigned... Is>
auto transform_tensor_to_4d(
    OpBackend& op,
    sh::graph& graph,
    const TensorsPair& tensor,
    std::optional<sh::tensor>& tensorStorageOpt) {
  if (tensor.pt_t.sizes().size() != 4) {
    tensorStorageOpt = get_4d_tensor(op, graph, tensor);
    return std::make_tuple(TensorDataGetter<Is>{}(*tensorStorageOpt)...);
  }
  return std::make_tuple(TensorDataGetter<Is>{}(tensor)...);
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

bool is_no_reshape_op(const OpBackend& op) {
  return op.GetGuid().find("batch_norm_inf_reshape") != std::string::npos ||
      op.GetGuid().find("_native_batch_norm_legit.no_stats") != std::string::npos;
}

std::vector<sh::tensor> handle_batch_norm_training_fwd(
    OpBackend& op,
    sh::graph& graph,
    const TensorsPair& input,
    const c10::optional<TensorsPair>& weight_opt,
    const c10::optional<TensorsPair>& bias_opt,
    const c10::optional<TensorsPair>& running_mean_opt,
    const c10::optional<TensorsPair>& running_var_opt,
    const std::shared_ptr<void>& params,
    const size_t params_size,
    const sizes_vec& out_shapes) {
  using namespace BNFwd;

  c10::IntArrayRef rm_size = get_rm_size(input.pt_t);
  std::optional<sh::tensor> weightStorageOpt;
  auto [weight] = get_or_create_tensor<TENSOR_IDX>(
      op,
      graph,
      weight_opt,
      rm_size,
      c10::ScalarType::Float,
      1,
      weightStorageOpt);
  weight = cast_if_necessary_or_default(
      &op, graph, weight_opt, weight, weightStorageOpt);
  std::optional<sh::tensor> biasStorageOpt;
  auto [bias] = get_or_create_tensor<TENSOR_IDX>(
      op, graph, bias_opt, rm_size, c10::ScalarType::Float, 0, biasStorageOpt);
  bias = cast_if_necessary_or_default(
      &op, graph, bias_opt, bias, biasStorageOpt);
  std::optional<sh::tensor> runningMeanStorageOpt;
  auto [running_mean, running_mean_storage_or_idx] =
      get_or_create_tensor<TENSOR_IDX, STORAGE_IDX>(
          op,
          graph,
          running_mean_opt,
          rm_size,
          c10::ScalarType::Float,
          0,
          runningMeanStorageOpt);
  running_mean = cast_if_necessary_or_default(
      &op, graph, running_mean_opt, running_mean, runningMeanStorageOpt);
  std::optional<sh::tensor> runningVarStorageOpt;
  auto [running_var, running_var_storage_or_idx] =
      get_or_create_tensor<TENSOR_IDX, STORAGE_IDX>(
          op,
          graph,
          running_var_opt,
          rm_size,
          c10::ScalarType::Float,
          get_running_var_def_value(op),
          runningVarStorageOpt);
  running_var = cast_if_necessary_or_default(
      &op, graph, running_var_opt, running_var, runningVarStorageOpt);
  bool is_functional = is_batch_norm_functional(op);

  std::vector<sh::tensor> bn_out;
  if (is_no_reshape_op(op)) {
    auto input_4d_shape = input.pt_t.sizes().vec();
    bn_out = OpBackend::BuildNode(
        &op,
        graph,
        {get_guid_with_precision("batch_norm_reshape_fwd", op.ScalarType()),
         {input.syn_t, bias, weight, running_mean, running_var},
         {NodeAttr::NodeOutputAttr{
              input_4d_shape, op.ScalarType(), c10::optional<int>(0)},
          NodeAttr::NodeOutputAttr{
              out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 1},
          NodeAttr::NodeOutputAttr{
              out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, 2},
          is_functional
              ? NodeAttr::
                    NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 3}
              : NodeAttr::
                    NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, c10::nullopt, DATA_TENSOR, syn_type_na, running_mean_storage_or_idx}, // SAVED_ISTD_IDX?!
          is_functional
              ? NodeAttr::
                    NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 4}
              : NodeAttr::
                    NodeOutputAttr{out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, c10::nullopt, DATA_TENSOR, syn_type_na, running_var_storage_or_idx}},
         params.get(),
         params_size});
  } else {
    std::optional<sh::tensor> inputStorageOpt;
    const auto [input_4d, input_4d_shape] =
        transform_tensor_to_4d<TENSOR_IDX, SHAPE_IDX>(
            op, graph, input, inputStorageOpt);
    const auto input_dim = input.pt_t.sizes().size();
    bn_out = OpBackend::BuildNode(
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
                    NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, c10::nullopt, DATA_TENSOR, syn_type_na, running_mean_storage_or_idx}, // SAVED_ISTD_IDX?!
          is_functional
              ? NodeAttr::
                    NodeOutputAttr{out_shapes[SAVED_MEAN_IDX], c10::ScalarType::Float, 4}
              : NodeAttr::
                    NodeOutputAttr{out_shapes[SAVED_ISTD_IDX], c10::ScalarType::Float, c10::nullopt, DATA_TENSOR, syn_type_na, running_var_storage_or_idx}},
         params.get(),
         params_size});
  }

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
    const TensorsPair& input,
    const c10::optional<TensorsPair>& weight_opt,
    const c10::optional<TensorsPair>& bias_opt,
    const c10::optional<TensorsPair>& running_mean_opt,
    const c10::optional<TensorsPair>& running_var_opt,
    const std::shared_ptr<void>& params,
    const size_t params_size,
    const sizes_vec& out_shapes) {
  using namespace BNFwd;

  std::vector<sh::tensor> bn_out;
  bn_out.reserve(5);
  c10::IntArrayRef rm_size = get_rm_size(input.pt_t);

  std::optional<sh::tensor> weightStorageOpt;
  const auto [weight] = get_or_create_tensor<TENSOR_IDX>(
      op,
      graph,
      weight_opt,
      rm_size,
      c10::ScalarType::Float,
      1,
      weightStorageOpt);
  std::optional<sh::tensor> biasStorageOpt;
  const auto [bias] = get_or_create_tensor<TENSOR_IDX>(
      op, graph, bias_opt, rm_size, c10::ScalarType::Float, 0, biasStorageOpt);
  std::optional<sh::tensor> runningMeanStorageOpt;
  const auto [running_mean] = get_or_create_tensor<TENSOR_IDX>(
      op,
      graph,
      running_mean_opt,
      rm_size,
      c10::ScalarType::Float,
      0,
      runningMeanStorageOpt);
  std::optional<sh::tensor> runningVarStorageOpt;
  auto [running_var] = get_or_create_tensor<TENSOR_IDX>(
      op,
      graph,
      running_var_opt,
      rm_size,
      c10::ScalarType::Float,
      get_running_var_def_value(op),
      runningVarStorageOpt);
  running_var = cast_if_necessary_or_default(
      &op, graph, running_var_opt, running_var, runningVarStorageOpt);

  bn_out.emplace_back(std::move(
      OpBackend::BuildNode(
          &op,
          graph,
          {get_guid_with_precision("batch_norm_inf_reshape", op.ScalarType()),
           {input.syn_t, bias, weight, running_mean, running_var},
           {{out_shapes[INPUT_IDX], op.ScalarType(), c10::optional<int>(0)}},
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

void moveLastOutputTensorAtFront(OpBackend& op) {
  auto& outputInfMeta = op.GetOutputInfMeta();
  auto output_tensor_idx = outputInfMeta.GetOutputTensor().size() - 1;
  auto output_tensor = outputInfMeta.GetOutputTensor(output_tensor_idx);
  outputInfMeta.RemoveOutput(output_tensor_idx);
  outputInfMeta.PushOutputTensorAtFront(output_tensor);
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

OutputMetaDataVector BatchNormBwdMeta(const at::Stack& stack) {
  using namespace BNBwd;
  auto input = stack_tensor(stack, INPUT_IDX);
  auto weightBiasShape = stack.at(WEIGHT_IDX).isTensor()
      ? stack_tensor(stack, WEIGHT_IDX).sizes().vec()
      : get_rm_size(input).vec();

  OutputMetaDataVector metaVec(3);
  metaVec[0].shape = input.sizes().vec();
  metaVec[0].dtype = input.scalar_type();
  for (int i = 1; i <= 2; i++) {
    metaVec[i].shape = weightBiasShape;
    metaVec[i].dtype = at::ScalarType::Float;
  }

  return metaVec;
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
  StackGetter stackGetter(this, stack, "BatchNormOpBackend::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto weightOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto biasOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto runningMeanOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto runningVarOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  bool training = stackGetter.getNextInput<bool>();
  stackGetter.getNextInput<double>(); // momentum
  stackGetter.getNextInput<double>(); // epsilon

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

  reshape_tensor(*this, graph, input.pt_t.sizes(), bnOut[0], ScalarType());

  if (isOutputInfMode()) {
    moveLastOutputTensorAtFront(*this);
  }

  // [SW-176505] Set allowPermutation=False as GC cannot handle
  // the case transpose -> reshape -> batchnorm
  // If not set, GC will add an extra transpose on batchnorm output
  if (input.pt_t.sizes().size() != 4) {
    bnOut[0].set_dont_allow_permute(true);
  }

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
  StackGetter stackGetter(this, stack, "BatchNormNoTrainingOpBackend::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto weightOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto biasOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto runningMeanOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto runningVarOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  stackGetter.getNextInput<double>(); // momentum
  stackGetter.getNextInput<double>(); // epsilon

  size_t paramsSize; // Will be initialized by below call
  const auto params = FillBatchNormNoTrainingFwdParams(stack, paramsSize);
  const auto outShapes = BatchNormFwdOutputShape(stack);

  auto inOutLayout = getSynapseLayout(input.pt_t.dim());

  SetSynapseLayouts(
      {inOutLayout,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE},
      {inOutLayout});

  std::vector<sh::tensor> bnOut =
      (is_training(false, runningMeanOpt.has_value())
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
  StackGetter stackGetter(this, stack, "BatchNormNoStatsOpBackend::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto weightOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto biasOpt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  bool training = stackGetter.getNextInput<bool>();
  stackGetter.getNextInput<double>(); // momentum
  stackGetter.getNextInput<double>(); // epsilon

  size_t paramsSize; // Will be initialized by below call
  const auto params = FillBatchNormNoStatsFwdParams(stack, paramsSize);
  const auto outShapes = BatchNormNoStatsFwdOutputShape(stack);

  auto inOutLayout = getSynapseLayout(input.pt_t.dim());

  SetSynapseLayouts(
      {inOutLayout,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE},
      {inOutLayout,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});

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

  // [SW-176505] Set allowPermutation=False as GC cannot handle
  // the case transpose -> reshape -> batchnorm
  // If not set, GC will add an extra transpose on batchnorm output
  if (input.pt_t.sizes().size() != 4) {
    bnOut[0].set_dont_allow_permute(true);
  }

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
  StackGetter stackGetter(this, stack, "BatchNormFwdOpBackend::AddNode");
  auto grad_out = stackGetter.getNextInput<TensorsPair>();
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto weight_opt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto running_mean_opt =
      stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto running_var_opt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto saved_mean_opt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto saved_istd_opt = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  bool training = stackGetter.getNextInput<bool>();
  double eps = stackGetter.getNextInput<double>();
  auto meta = BatchNormBwdMeta(stack);
  /* 2. Perform frontend operations */
  // In case of batch norm:
  // 2.1 Preprocess inputs
  // 2.1.1 Reshape input to 4D

  std::optional<sh::tensor> inputStorageOpt;
  const auto [input_4d, input_4d_shape] =
      transform_tensor_to_4d<TENSOR_IDX, SHAPE_IDX>(
          *this, graph, input, inputStorageOpt);

  std::optional<sh::tensor> gradStorageOpt;
  const auto [grad_out_4d] = transform_tensor_to_4d<TENSOR_IDX>(
      *this, graph, grad_out, gradStorageOpt);

  c10::IntArrayRef rm_size = get_rm_size(input.pt_t);
  std::optional<sh::tensor> weightStorageOpt;
  auto [weight] = get_or_create_tensor<TENSOR_IDX>(
      *this,
      graph,
      weight_opt,
      rm_size,
      c10::ScalarType::Float,
      1,
      weightStorageOpt);

  synTensor saved_mean, saved_istd;
  std::optional<sh::tensor> saved_mean_storage, saved_istd_storage,
      weight_storage;
  if (!is_training(training, running_mean_opt.has_value())) {
    std::optional<sh::tensor> runningMeanStorageOpt;
    const auto [running_mean] = get_or_create_tensor<TENSOR_IDX>(
        *this,
        graph,
        running_mean_opt,
        rm_size,
        c10::ScalarType::Float,
        0,
        runningMeanStorageOpt);
    std::optional<sh::tensor> runningVarStorageOpt;
    const auto [running_var, running_var_shape] =
        get_or_create_tensor<TENSOR_IDX, SHAPE_IDX>(
            *this,
            graph,
            running_var_opt,
            rm_size,
            c10::ScalarType::Float,
            1,
            runningVarStorageOpt);

    // TODO calculate saved_mean, saved_istd
    saved_mean = running_mean;

    // istd = rsqrt(add(running_var, eps));
    auto eps_constant = ConstantHelper(graph, eps, meta[1].dtype, {1});

    auto rv_add_eps = std::move(BuildOp(
                                    graph,
                                    "add_fwd_f32",
                                    {running_var, eps_constant.get()},
                                    {{running_var_shape, meta[1].dtype}})
                                    .at(0));

    saved_istd_storage = std::move(BuildOp(
                                       graph,
                                       "rsqrt_fwd_f32",
                                       {rv_add_eps.get()},
                                       {{running_var_shape, meta[1].dtype}})
                                       .at(0));
    saved_istd = (*saved_istd_storage).get();
  } else {
    saved_mean = cast_if_necessary_or_default(
        this,
        graph,
        saved_mean_opt,
        saved_mean_opt.has_value() ? (*saved_mean_opt).syn_t : saved_mean,
        saved_mean_storage);

    saved_istd = cast_if_necessary_or_default(
        this,
        graph,
        saved_istd_opt,
        saved_istd_opt.has_value() ? (*saved_istd_opt).syn_t : saved_istd,
        saved_istd_storage);
  }
  weight = cast_if_necessary_or_default(
      this, graph, weight_opt, weight, weight_storage);

  size_t size; // Will be initialized by below call
  const auto params = FillBatchNormBwdParams(stack, size);

  c10::optional<int> final_result_index_0 =
      meta[INPUT_GRAD_IDX].shape.size() != 4
      ? c10::optional<int>{c10::nullopt}
      : c10::optional<int>{INPUT_GRAD_IDX};
  auto bn_out = BuildOp(
      graph,
      get_guid_with_precision("batch_norm_bwd", meta[0].dtype),
      {input_4d, grad_out_4d, saved_mean, saved_istd, weight},
      {{input_4d_shape, meta[INPUT_GRAD_IDX].dtype, final_result_index_0},
       {meta[BIAS_GRAD_IDX].shape, meta[BIAS_GRAD_IDX].dtype, BIAS_GRAD_IDX},
       {meta[WEIGHT_GRAD_IDX].shape,
        meta[WEIGHT_GRAD_IDX].dtype,
        WEIGHT_GRAD_IDX}},
      params.get(),
      size);

  // 2.4 Postprocess outputs
  // 2.4.1 Reshape output to original input's shape
  reshape_tensor(
      *this,
      graph,
      meta[INPUT_GRAD_IDX].shape,
      bn_out[INPUT_GRAD_IDX],
      meta[INPUT_GRAD_IDX].dtype);

  // [SW-176505] Set allowPermutation=False as GC cannot handle
  // the case transpose -> reshape -> batchnorm
  // If not set, GC will add an extra transpose on batchnorm output
  if ((meta[INPUT_GRAD_IDX].shape).size() != 4) {
    bn_out[INPUT_GRAD_IDX].set_dont_allow_permute(true);
  }

  syn_out(INPUT_GRAD_IDX) = std::move(bn_out[0]);
  syn_out(WEIGHT_GRAD_IDX) = std::move(bn_out[2]);
  syn_out(BIAS_GRAD_IDX) = std::move(bn_out[1]);
}

} // namespace habana
