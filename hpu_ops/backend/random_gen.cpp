/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/bernoulli.h"
#include "generated/backend/normal.h"
#include "generated/backend/poisson.h"
#include "generated/backend/random.h"
#include "generated/backend/uniform.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/habana_random_ops.h"

namespace habana {
const unsigned SIZE_INDEX = 2;
const unsigned DTYPE_INDEX = 4;
const unsigned LAYOUT_INDEX = 5;

enum NormalVariant {
  NORMAL_FF = 0,
  NORMAL_TF = 1,
  NORMAL_FT = 2,
  NORMAL_TT = 3
};

OutputMetaDataVector NormalMeta(const at::Stack& stack) {
  OutputMetaData meta;
  c10::ScalarType dtype;
  if (stack.size() > DTYPE_INDEX) {
    dtype = stack.at(DTYPE_INDEX)
                .toOptional<at::ScalarType>()
                .value_or(at::get_default_dtype_as_scalartype());
  } else if (stack.at(0).isTensor() && stack.at(1).isTensor()) {
    dtype = at::result_type(stack.at(0).toTensor(), stack.at(1).isTensor());
  } else if (stack.at(0).isTensor() && !stack.at(1).isTensor()) {
    dtype = stack.at(0).toTensor().scalar_type();
  } else if (!stack.at(0).isTensor() && stack.at(1).isTensor()) {
    dtype = stack.at(1).toTensor().scalar_type();
  } else {
    dtype = at::get_default_dtype_as_scalartype();
  }

  meta.dtype = dtype;
  std::vector<int64_t> sz;
  if (stack.at(0).isTensor()) {
    sz = stack.at(0).toTensor().sizes().vec();
  } else if (stack.at(1).isTensor()) {
    sz = stack.at(1).toTensor().sizes().vec();
  } else {
    sz = stack.at(SIZE_INDEX).toIntVector();
    meta.layout = stack.at(LAYOUT_INDEX)
                      .toOptional<at::Layout>()
                      .value_or(at::Layout::Strided);
  }
  meta.shape = sz;
  return {meta};
}

static std::shared_ptr<void> RandomUniformParams(
    at::ScalarType type,
    at::optional<float> from,
    at::optional<float> to,
    size_t& size) {
  PARAMS_STUB(ns_RandomUniform::Params);
  /*
  NOTE: As per PyTorch specification, for floating point types, if unspecified,
  range will be [0, 2^mantissa] to ensure that every value is representable. For
  example, torch.tensor(1, dtype=torch.double).random_() will be uniform in [0,
  2^53].
  */
  switch (type) {
    case at::ScalarType::Float:
    case at::ScalarType::Double:
      params->low = from.has_value() ? *from : 0;
      params->high = to.has_value() ? *to : 1 << 24;
      break;
    case at::ScalarType::BFloat16:
      params->low = from.has_value() ? *from : 0;
      params->high = to.has_value() ? *to : 1 << 8;
      break;
    case at::ScalarType::Int:
      params->low = from.has_value() ? *from : 0;
      params->high = to.has_value()
          ? *to
          : static_cast<float>(std::numeric_limits<int>::max());
      break;
    default:
      TORCH_CHECK(false, "Got unsupported type for random uniform: ", type);
      break;
  }

  PT_KERNEL_DEBUG(__func__, " low: ", params->low, " high: ", params->high);

  return params;
}

std::shared_ptr<void> FillRandomParams(const at::Stack& stack, size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(), c10::nullopt, c10::nullopt, size);
}

std::shared_ptr<void> FillRandomFromParams(
    const at::Stack& stack,
    size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      stack.at(1).isNone() ? c10::nullopt
                           : c10::make_optional<float>(stack.at(1).toInt()),
      c10::make_optional<float>(stack.at(2).toInt()),
      size);
}

std::shared_ptr<void> FillRandomToParams(const at::Stack& stack, size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      c10::nullopt,
      c10::make_optional<float>(stack.at(1).toInt()),
      size);
}

std::shared_ptr<void> FillUniformParams(const at::Stack& stack, size_t& size) {
  return RandomUniformParams(
      stack_tensor(stack, 0).scalar_type(),
      c10::make_optional<float>(stack.at(1).toDouble()),
      c10::make_optional<float>(stack.at(2).toDouble()),
      size);
}

std::shared_ptr<void> FillNormal2Params(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_RandomNormal::Params);
  params->mean = static_cast<float>(stack.at(0).toDouble());
  params->stddev = static_cast<float>(stack.at(1).toDouble());
  return params;
}

synapse_helpers::tensor NormalTensorHelper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    synTensor syn_in0,
    synTensor syn_in1,
    synTensor syn_seed,
    NormalVariant normal_variant) {
  const auto meta = NormalMeta(stack)[0];
  std::vector<synTensor> inputs;
  /*
  NOTE: Due to TPC guid limitations of random_normal.
  We will use the linear transformation approach N(mean, std) = (N(0,1) +
  mean)*std
  */
  inputs.push_back(nullptr);
  inputs.push_back(syn_seed);
  size_t size = 0;
  PARAMS_STUB(ns_RandomNormal::Params);
  params->mean = static_cast<float>(0.); // mean;
  params->stddev = static_cast<float>(1.); // stddev;
  op->CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  auto normal = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("random_normal_fwd", meta.dtype),
       inputs,
       {{meta.shape, meta.dtype}},
       params.get(),
       size});
  if (normal_variant == NORMAL_TF) {
    // insert mulOp if necessary
    float stddev = static_cast<float>(stack.at(1).toDouble());
    if (stddev != 1.0) {
      auto stddev_tensor =
          OpBackend::BuildConstant(op, graph, stddev, meta.dtype, {1});
      auto mulOp = OpBackend::BuildNode(
          op,
          graph,
          {get_guid_with_precision("mult_fwd", meta.dtype),
           {stddev_tensor.get(), normal[0].get()},
           {{meta.shape, meta.dtype}}});
      auto addOp = OpBackend::BuildNode(
          op,
          graph,
          {get_guid_with_precision("add_fwd", meta.dtype),
           {syn_in0, mulOp[0].get()},
           {{meta.shape, meta.dtype, 0}}});

      return std::move(addOp[0]);
    } else {
      auto addOp = OpBackend::BuildNode(
          op,
          graph,
          {get_guid_with_precision("add_fwd", meta.dtype),
           {syn_in0, normal[0].get()},
           {{meta.shape, meta.dtype, 0}}});
      return std::move(addOp[0]);
    }
  } else if (normal_variant == NORMAL_FT) {
    // insert addOp if necessary
    float mean = static_cast<float>(stack.at(0).toDouble());
    if (mean != 0.0) {
      auto mulOp = OpBackend::BuildNode(
          op,
          graph,
          {get_guid_with_precision("mult_fwd", meta.dtype),
           {syn_in0,
            normal[0].get()}, // in this variant syn_in0 is stddev tensor
           {{meta.shape, meta.dtype}}});
      auto mean_tensor =
          OpBackend::BuildConstant(op, graph, mean, meta.dtype, {1});
      auto addOp = OpBackend::BuildNode(
          op,
          graph,
          {get_guid_with_precision("add_fwd", meta.dtype),
           {mean_tensor.get(), mulOp[0].get()},
           {{meta.shape, meta.dtype, 0}}});
      return std::move(addOp[0]);
    } else {
      auto mulOp = OpBackend::BuildNode(
          op,
          graph,
          {get_guid_with_precision("mult_fwd", meta.dtype),
           {syn_in0,
            normal[0].get()}, // in this variant syn_in0 is stddev tensor
           {{meta.shape, meta.dtype, 0}}});
      return std::move(mulOp[0]);
    }
  } else {
    auto mulOp = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("mult_fwd", meta.dtype),
         {syn_in1, normal[0].get()},
         {{meta.shape, meta.dtype}}});
    auto addOp = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("add_fwd", meta.dtype),
         {syn_in0, mulOp[0].get()},
         {{meta.shape, meta.dtype, 0}}});
    return std::move(addOp[0]);
  }
}

synapse_helpers::tensor NormalFloatFloatHelper(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    synTensor syn_seed) {
  const auto meta = NormalMeta(stack)[0];
  std::vector<synTensor> inputs;
  auto mean = stack.at(0).toDouble();
  auto stddev = stack.at(1).toDouble();
  // Input tensors to random_normal_fwd are stddev and seed tensors
  inputs.push_back(nullptr);
  inputs.push_back(syn_seed);
  size_t size = 0;
  PARAMS_STUB(ns_RandomNormal::Params);
  params->mean = static_cast<float>(mean);
  params->stddev = static_cast<float>(stddev);
  op->CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  auto normal = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("random_normal_fwd", meta.dtype),
       inputs,
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size});
  return std::move(normal[0]);
}

void NormalBE::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto meta = NormalMeta(stack)[0];
  std::vector<synTensor> inputs;
  // For eager mode, the normal.float_float gets decomposed to normal_ and
  // doesn't come here. So, we can make the safe assumption that seed tensor is
  // the last element in the stack.
  // create the correct enum
  NormalVariant normal_variant =
      (NormalVariant)(stack.at(0).isTensor() + stack.at(1).isTensor() * 2);
  synTensor syn_seed_t;
  if (stack.at(stack.size() - 1).isTensor()) { // eager mode
    if (normal_variant == NORMAL_TT) {
      syn_seed_t = syn_in(2); // tensors = mean, stddev, seed
    } else {
      syn_seed_t = syn_in(1); // TF or FT case - tensors = mean or stddev, seed
    }
  } else {
    if ((normal_variant == NORMAL_FF) && stack.at(3).isTensor()) {
      syn_seed_t = syn_in(0); // eager mode with Gen replaced by seed
    } else {
      syn_seed_t = syn_seed(); // torch.compile mode
    }
  }
  if (normal_variant != NORMAL_FF) {
    syn_out(0) = NormalTensorHelper(
        this,
        graph,
        stack,
        syn_in(0),
        (normal_variant != NORMAL_TF)
            ? ((normal_variant == NORMAL_TT) ? syn_in(1) : syn_in(0))
            : nullptr,
        syn_seed_t,
        normal_variant);
  } else {
    syn_out(0) = NormalFloatFloatHelper(this, graph, stack, syn_seed_t);
  }
  return;
}

void RandomSeedTensorInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = stack_tensor(stack, 0).sizes();
  // The following kernels have an optional tesor input before seed tensor
  // (also optional) input. Eg stddev tensor. If we are not passing this tensor
  // to TPC we should set it as null. This is because TPC requires that all
  // leading unused optional tensors are passed as null if any valid tensor
  // (eg. seed in this case) follows them.
  static std::vector<std::string> guids_seed_tensor_pos_check = {
      "random_normal", // TPC spec tensor list : {stddev(opt), seed(opt)}
      "log_normal"}; // TPC spec tensor list : {stddev(opt), seed(opt)}
  std::vector<synTensor> inputs;

  for (size_t i = 0; i < guids_seed_tensor_pos_check.size(); i++) {
    if (guid_.find(guids_seed_tensor_pos_check[i]) != std::string::npos) {
      inputs.push_back(nullptr);
      break;
    }
  }

  inputs.push_back(syn_in(1)); // insert seed tensor
  CreateShapeTensorInput(
      graph,
      ScalarType() == c10::ScalarType::Int ? at::kFloat : ScalarType(),
      outshape,
      inputs);
  size_t size = 0;
  auto rand_params = FillParams(stack, size);

  if (ScalarType() == c10::ScalarType::Int) {
    auto rand = BuildOp(
        graph,
        update_guid_dtype(guid_, "f32"),
        std::move(inputs),
        {{outshape}},
        rand_params.get(),
        size);

    PARAMS_STUB(ns_CastKernel::Params);
    // Round down so that the upper limit is not included in the generated seq.
    // The assumption is that the float vaues dont include the upper limit.
    params->round_mode = CAST_ROUND_DOWN;
    auto cast = BuildOp(
        graph,
        "cast_f32_to_i32",
        {rand[0].get()},
        {{outshape, ScalarType(), 0}},
        params.get(),
        size);
    syn_out(0) = std::move(cast[0]);
    return;
  }

  auto rand = BuildOp(
      graph,
      guid_,
      std::move(inputs),
      {{outshape, ScalarType(), 0}},
      rand_params.get(),
      size);
  syn_out(0) = std::move(rand[0]);
}

OutputMetaDataVector HabanaRandOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack[0].toIntVector();
  meta.dtype =
      stack[2].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);
  meta.layout = stack[3].toOptional<at::Layout>().value_or(at::kStrided);
  return {meta};
}

HabanaRand::HabanaRand(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "random_uniform", scalar_type, {1}, {}, {}, false) {
  SetOutputMetaFn(HabanaRandOutputMeta);
}

void HabanaRand::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = stack[0].toIntVector();
  auto dtype =
      stack[2].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);

  ns_RandomUniform::Params params{};
  params.low = 0.0;
  params.high = 1.0;

  std::vector<synTensor> inputs{syn_in(0)};

  CreateShapeTensorInput(graph, dtype, output_shape, inputs);

  auto rand = BuildOp(
      graph,
      get_guid_with_precision("random_uniform", dtype),
      std::move(inputs),
      {{output_shape, dtype, 0}},
      &params,
      sizeof(params));
  syn_out(0) = std::move(rand[0]);
}

HabanaRandn::HabanaRandn(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "random_normal", scalar_type, {1}, {}, {}, false) {
  SetOutputMetaFn(HabanaRandOutputMeta);
}

void HabanaRandn::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto output_shape = stack[0].toIntVector();
  auto seed = stack_tensor(stack, 1);
  auto dtype =
      stack[2].toOptional<at::ScalarType>().value_or(at::ScalarType::Float);

  ns_RandomNormal::Params params{};
  params.mean = 0.0;
  params.stddev = 1.0;

  std::vector<synTensor> inputs{nullptr, syn_in(0)};

  CreateShapeTensorInput(graph, dtype, output_shape, inputs);

  auto rand = BuildOp(
      graph,
      get_guid_with_precision("random_normal", dtype),
      std::move(inputs),
      {{output_shape, dtype, 0}},
      &params,
      sizeof(params));
  syn_out(0) = std::move(rand[0]);
}

OutputMetaDataVector HabanaSeedGeneratorOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = {stack[2].toInt()};
  meta.dtype = at::ScalarType::Int;
  return {meta};
}

HabanaSeedGenerator::HabanaSeedGenerator(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "philox_random_uniform",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(HabanaSeedGeneratorOutputMeta);
}

void HabanaSeedGenerator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  ns_PhiloxRandomUniform::ParamsV3 params{};
  params.low_i = 0;
  params.high_i = std::numeric_limits<int32_t>::max();
  auto philox_dtype = at::ScalarType::Int;

  std::vector<int64_t> output_shape{stack[2].toInt()};

  std::vector<synTensor> inputs{syn_in(0), syn_in(1)};

  CreateShapeTensorInput(graph, philox_dtype, output_shape, inputs);

  auto rand = BuildOp(
      graph,
      get_guid_with_precision("philox_random_uniform", philox_dtype),
      std::move(inputs),
      {{output_shape, philox_dtype, 0}},
      &params,
      sizeof(params));

  syn_out(0) = std::move(rand[0]);
}
} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::habana_rand", KERNEL_FN_GLOBAL(habana::HabanaRand))
        .add("hpu::habana_randn", KERNEL_FN_GLOBAL(habana::HabanaRandn))
        .add(
            "hpu::habana_seed_generator",
            KERNEL_FN_GLOBAL(habana::HabanaSeedGenerator));
