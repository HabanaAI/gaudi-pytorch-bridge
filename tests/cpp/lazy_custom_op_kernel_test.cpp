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
#include <gtest/gtest.h>
#include <perf_lib_layer_params.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include "backend/habana_operator.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"
#include "hpu_ops/hpu_op_helper.h"
#include "include/habanalabs/hpu_custom_op.h"

using namespace habana_lazy;
using namespace at;

class LazyCustomKernelKernelTest : public habana_lazy_test::LazyTest {
 public:
  LazyCustomKernelKernelTest() {
    register_custom_add();
    register_custom_gelu();
    register_custom_topk();
  }

 private:
  void register_custom_add() {
    // Registering ustom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc};
    // output desc
    habana::custom_op::OutputDesc output_desc{0};
    std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_add",
        "add_fwd_f32",
        inputs_desc,
        outputs_desc,
        nullptr);
  }

  void register_custom_gelu() {
    // Registering custom_op::custom_gelu
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    std::vector<habana::custom_op::InputDesc> inputs_desc{input_a_desc};
    // output desc
    habana::custom_op::OutputDesc output_desc{0};
    habana::custom_op::OutputDesc output_desc_2{1};
    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc, output_desc_2};
    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_gelu",
        "gelu_fwd_f32",
        inputs_desc,
        outputs_desc,
        nullptr);
  }

  void register_custom_topk() {
    // Registering ustom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::USER_PARAMS, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::USER_PARAMS, 2};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::USER_PARAMS, 3};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc, input_d_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      auto k = inputs[1].toInt(); // k
      auto dim = inputs[2].toInt(); // dim
      std::vector<int64_t> result_sizes = self.sizes().vec();
      if (result_sizes.size() > 0) {
        result_sizes[dim] = k;
      }
      return result_sizes;
    };
    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};
    habana::custom_op::OutputDesc output_desc_2{
        1, c10::ScalarType::Long, output_size_lambda};
    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc, output_desc_2};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(synBeamParams);
      auto self = inputs[0].toTensor(); // input
      params->bsw = inputs[1].toInt(); // k
      auto dim = inputs[2].toInt(); // axis
      params->axis = habana::get_dim_in_tpc_order(dim, self.dim());
      params->bottomK = inputs[3].toBool(); // bottom
      return params;
    };

    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_topk",
        "topk",
        inputs_desc,
        outputs_desc,
        user_params_lambda);
  }
};

at::Tensor custom_add_execute(torch::Tensor input_a, torch::Tensor input_b) {
  std::vector<c10::IValue> inputs{input_a, input_b};
  auto op_desc = habana::KernelRegistry().get_legacy_user_custom_op_desc(
      "custom_op::custom_add");
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  return output[0];
}

at::Tensor custom_gelu_execute(torch::Tensor input_a) {
  std::vector<c10::IValue> inputs{input_a};
  auto op_desc = habana::KernelRegistry().get_legacy_user_custom_op_desc(
      "custom_op::custom_gelu");
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  return output[0];
}

std::tuple<at::Tensor, at::Tensor> custom_topk_execute(
    torch::Tensor input_a,
    at::Scalar k,
    at::Scalar axis,
    bool bottom) {
  std::vector<c10::IValue> inputs{input_a, k, axis, bottom};
  auto op_desc = habana::KernelRegistry().get_legacy_user_custom_op_desc(
      "custom_op::custom_topk");
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  return {output[0], output[1]};
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_add(Tensor self, Tensor other) -> Tensor");
  m.def("custom_gelu(Tensor self) -> Tensor");
  m.def(
      "custom_topk(Tensor self, Scalar k, Scalar axis, bool bottom) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_add", custom_add_execute);
  m.impl("custom_gelu", custom_gelu_execute);
  m.impl("custom_topk", custom_topk_execute);
}

TEST_F(LazyCustomKernelKernelTest, BinaryOp) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  SetSeed();
  torch::Tensor input_a_cpu = torch::randn({2, 2}, torch::dtype(torch::kFloat));
  torch::Tensor input_b_cpu = torch::randn({2, 2}, torch::dtype(torch::kFloat));

  torch::Tensor results_cpu = input_a_cpu.add(input_b_cpu);

  torch::Tensor input_a = input_a_cpu.to(torch::kHPU);
  torch::Tensor input_b = input_b_cpu.to(torch::kHPU);

  at::Tensor result = custom_add_execute(input_a, input_b);
  auto hl_result = SyncAndGetHbLazyTensor(result);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  std::vector<at::Tensor> input_list{input_a, input_b};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check("custom_op::custom_add")
      ->run(*hlexec->get_graph());

  bool equal = results_cpu.allclose(result.to(torch::kCPU), 0, 0);
  EXPECT_TRUE(equal);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyCustomKernelKernelTest, MultipleOutputs) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  SetSeed();
  torch::Tensor input_a_cpu = torch::randn({2, 2}, torch::dtype(torch::kFloat));
  torch::Tensor input_a = input_a_cpu.to(torch::kHPU);

  torch::Tensor results_cpu = torch::nn::functional::gelu(input_a);

  at::Tensor result = custom_gelu_execute(input_a);
  auto hl_result = SyncAndGetHbLazyTensor(result);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  std::vector<at::Tensor> input_list{input_a};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check("custom_op::custom_gelu")
      ->run(*hlexec->get_graph());

  bool equal = results_cpu.allclose(result.to(torch::kCPU), 0.5, 0.5);
  EXPECT_TRUE(equal);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyCustomKernelKernelTest, ShapeInference) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  torch::Tensor input_cpu = torch::randn({6, 6}, torch::dtype(torch::kFloat));
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  int k = 3;
  auto results_cpu = input_cpu.topk(k);
  auto results_habana = custom_topk_execute(input_hpu, k, 1, false);

  bool equal = std::get<0>(results_cpu)
                   .allclose(std::get<0>(results_habana).to(torch::kCPU), 0, 0);
  bool equal_indices =
      std::get<1>(results_cpu)
          .allclose(std::get<1>(results_habana).to(torch::kCPU), 0, 0);
  EXPECT_TRUE(equal);
  EXPECT_TRUE(equal_indices);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}
