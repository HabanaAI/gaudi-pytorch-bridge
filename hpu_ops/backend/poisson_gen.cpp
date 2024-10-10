/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/poisson.h"
#include "hpu_ops/habana_random_ops.h"

namespace habana {
std::shared_ptr<void> FillPoissonParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_RandomPoisson::Params);
  params->lambda = 0.0;
  params->poissonFlavor = RandomPoissonFlavor_t::WITH_DIST;
  return params;
}

HabanaPoisson::HabanaPoisson(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "habana_poisson", scalar_type, {1}, {}, {}, false) {}

void HabanaPoisson::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& input_tensor = stack_tensor(stack, 1);
  const auto& dtype = input_tensor.scalar_type();
  std::vector<synTensor> inputs = {syn_in(1), syn_in(0)};

  size_t params_size = 0;
  const auto& params = FillPoissonParams(stack, params_size);
  auto poisson = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("random_poisson_fwd", dtype),
       inputs,
       {{input_tensor.sizes().vec(), dtype, 0}},
       params.get(),
       params_size});

  syn_out(0) = std::move(poisson[0]);
}

HabanaPoissonCheckpoint::HabanaPoissonCheckpoint(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "habana_poisson",
          scalar_type,
          {0, 1},
          {},
          {},
          false) {}

void HabanaPoissonCheckpoint::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto seed =
      BuildOp(graph, "identity", {syn_in(0)}, {{{}, at::ScalarType::Int, 0}});
  syn_out(0) = std::move(seed[0]);

  const auto& input_tensor = stack_tensor(stack, 1);
  const auto& dtype = input_tensor.scalar_type();
  std::vector<synTensor> inputs = {syn_in(1), syn_in(0)};

  size_t params_size = 0;
  const auto& params = FillPoissonParams(stack, params_size);
  auto poisson = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("random_poisson_fwd", dtype),
       inputs,
       {{input_tensor.sizes().vec(), dtype, 1}},
       params.get(),
       params_size});

  syn_out(1) = std::move(poisson[0]);
}
} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry().REGISTER_RANDOM_OP(poisson, Poisson);
