/**
 * Copyright (c) 2024-2025 Intel Corporation
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

#include "shared_layer_report_generator.h"
#include "executor.h"
#include "generated_validator_headers.h"
#include "stack_generator.h"

namespace slrg {
void SharedLayerReportGenerator::register_exceptions() {
  register__adaptive_avg_pool2d_exception();
  register_abs__exception();
  register_bmm_exception();
  register_bmm_out_exception();
  register_channel_shuffle_exception();
  register_clamp_exception();
  register_ctc_loss_custom_exception();
  register_ctc_loss_custom_backward_exception();
  register_ctc_loss_exception();
  register_ctc_loss_tensor_exception();
  register_fused_clip_norm_exception();
  register_grid_sample_exception();
  register_im2col_exception();
  register_im2col_out_exception();
  register_index_reduce__exception();
  register_in_place_interleave_exception();
  register_kv_reorder_exception();
  register_linear_exception();
  register_masked_fill_exception();
  register_masked_scatter_exception();
  register_max_pool2d_exception();
  register_max_pool3d_exception();
  register_mm_exception();
  register_mm_out_exception();
  register_multi_margin_loss_exception();
  register_multi_margin_loss_out_exception();
  register_multilabel_margin_loss_exception();
  register_nll_loss_forward_exception();
  register_nll_loss_forward_output_exception();
  register_optimizer_resource_apply_momentum_exception();
  register_reflection_pad_exception();
  register_replication_pad_exception();
  register_rotary_pos_embedding_exception();
  register_scaled_triangular_softmax_retain_exception();
  register_scatter_add__exception();
  register_scatter_exception();
  register_scatter_out_exception();
  register_searchsorted_exception();
  register_max_unpool2d_exception();
  register_max_unpool2d_out_exception();
  register_max_unpool3d_exception();
  register_max_unpool3d_out_exception();
  register_upsample_exception();
  register_upsample_out_exception();
  register_where_exception();
  register_where_out_exception();
  register_static_exceptions();
}

void SharedLayerReportGenerator::register__adaptive_avg_pool2d_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[2] output_size", "_adaptive_avg_pool2d"));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator__adaptive_avg_pool2d));

  register_op(
      {/* op_name */ "_adaptive_avg_pool2d",
       /* overload */ "",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_abs__exception() {
  custom_stack_generators.push_back(
      std::make_unique<SchemaStackGenerator>("Tensor self", "abs_"));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_abs_));

  register_op(
      {/* op_name */ "abs_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_bmm_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, Tensor mat2", "bmm", "", std::vector<std::int64_t>{3}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_bmm));

  register_op(
      {/* op_name */ "bmm", /* overload */ "", /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "bmm",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_bmm_out_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, Tensor mat2, Tensor out",
      "bmm",
      "bmm.out",
      std::vector<std::int64_t>{3}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_bmm_out));

  register_op(
      {/* op_name */ "bmm", /* overload */ "out", /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_channel_shuffle_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt groups",
      "channel_shuffle",
      "",
      std::vector<std::int64_t>{3}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_channel_shuffle));

  register_op(
      {/* op_name */ "channel_shuffle",
       /* overload */ "",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "ChannelShuffle",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "channel_shuffle",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_clamp_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "min",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ true,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ false,
               /* ranks */ std::vector<int64_t>{1},
               /* match_rank */ false,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "max",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ true,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ true,
               /* ranks */ std::vector<int64_t>{1},
               /* match_rank */ false,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "clamp")));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_clamp));

  register_op(
      {/* op_name */ "clamp", /* overload */ "", /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "clamp",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_ctc_loss_custom_exception() {
  custom_stack_generators.push_back(std::make_unique<
                                    StackGenerator>(StackGenerator(
      {
          InputDescriptor{
              /* name */ "log_probs",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ true,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "targets",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
              /* match_precision_type */ false,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "input_lengths",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
              /* match_precision_type */ false,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "target_lengths",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
              /* match_precision_type */ false,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "blank",
              /* type */ InputType::NATIVE_INT,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{0},
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "reduction",
              /* type */ InputType::NATIVE_INT,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{0},
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "zero_infinity",
              /* type */ InputType::NATIVE_BOOL,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{false},
              /* is_array */ false,
              /* array_length */ std::nullopt},
      },
      /* blacklisted_precision_types */ {},
      /* whitelisted_precision_types */ {},
      "ctc_loss_custom")));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_ctc_loss_custom));

  register_op(
      {/* op_name */ "ctc_loss_custom",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_ctc_loss_custom_backward_exception() {
  custom_stack_generators.push_back(std::make_unique<
                                    StackGenerator>(StackGenerator(
      {
          InputDescriptor{
              /* name */ "grad",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ true,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "log_probs",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ true,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "targets",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
              /* match_precision_type */ false,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "input_lengths",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
              /* match_precision_type */ false,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "target_lengths",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
              /* match_precision_type */ false,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "neg_log_likelihood",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ true,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "log_alpha",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ true,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "blank",
              /* type */ InputType::NATIVE_INT,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{0},
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "reduction",
              /* type */ InputType::NATIVE_INT,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{0},
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "zero_infinity",
              /* type */ InputType::NATIVE_BOOL,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{false},
              /* is_array */ false,
              /* array_length */ std::nullopt},
      },
      /* blacklisted_precision_types */ {},
      /* whitelisted_precision_types */ {},
      "ctc_loss_custom_backward")));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_ctc_loss_custom_backward));

  register_op(
      {/* op_name */ "ctc_loss_custom_backward",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_ctc_loss_exception() {
  custom_stack_generators.push_back(std::make_unique<
                                    StackGenerator>(StackGenerator(
      {
          InputDescriptor{
              /* name */ "log_probs",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ true,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "targets",
              /* type */ InputType::PT_TENSOR,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ true,
              /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
              /* match_precision_type */ false,
              /* values */ std::nullopt,
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "input_lengths",
              /* type */ InputType::NATIVE_INT,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{1},
              /* is_array */ true,
              /* array_length */ 1},
          InputDescriptor{
              /* name */ "target_lengths",
              /* type */ InputType::NATIVE_INT,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{1},
              /* is_array */ true,
              /* array_length */ 1},
          InputDescriptor{
              /* name */ "blank",
              /* type */ InputType::NATIVE_INT,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{0},
              /* is_array */ false,
              /* array_length */ std::nullopt},
          InputDescriptor{
              /* name */ "zero_infinity",
              /* type */ InputType::NATIVE_BOOL,
              /* is_optional */ false,
              /* allow_only_none */ std::nullopt,
              /* allow_none */ std::nullopt,
              /* ranks */ std::nullopt,
              /* match_rank */ std::nullopt,
              /* dtypes */ std::nullopt,
              /* match_precision_type */ std::nullopt,
              /* values */ std::vector<std::any>{false},
              /* is_array */ false,
              /* array_length */ std::nullopt},
      },
      /* blacklisted_precision_types */ {},
      /* whitelisted_precision_types */ {},
      "_ctc_loss",
      "",
      {2})));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator__ctc_loss));

  register_op(
      {/* op_name */ "_ctc_loss",
       /* overload */ "",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_ctc_loss_tensor_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {
              InputDescriptor{
                  /* name */ "log_probs",
                  /* type */ InputType::PT_TENSOR,
                  /* is_optional */ false,
                  /* allow_only_none */ std::nullopt,
                  /* allow_none */ std::nullopt,
                  /* ranks */ std::nullopt,
                  /* match_rank */ true,
                  /* dtypes */ std::nullopt,
                  /* match_precision_type */ true,
                  /* values */ std::nullopt,
                  /* is_array */ false,
                  /* array_length */ std::nullopt},
              InputDescriptor{
                  /* name */ "targets",
                  /* type */ InputType::PT_TENSOR,
                  /* is_optional */ false,
                  /* allow_only_none */ std::nullopt,
                  /* allow_none */ std::nullopt,
                  /* ranks */ std::nullopt,
                  /* match_rank */ true,
                  /* dtypes */
                  std::vector<c10::ScalarType>{c10::ScalarType::Int},
                  /* match_precision_type */ false,
                  /* values */ std::nullopt,
                  /* is_array */ false,
                  /* array_length */ std::nullopt},
              InputDescriptor{
                  /* name */ "input_lengths",
                  /* type */ InputType::PT_TENSOR,
                  /* is_optional */ false,
                  /* allow_only_none */ std::nullopt,
                  /* allow_none */ std::nullopt,
                  /* ranks */ std::vector<int64_t>{1},
                  /* match_rank */ false,
                  /* dtypes */
                  std::vector<c10::ScalarType>{c10::ScalarType::Int},
                  /* match_precision_type */ false,
                  /* values */ std::nullopt,
                  /* is_array */ false,
                  /* array_length */ std::nullopt},
              InputDescriptor{
                  /* name */ "target_lengths",
                  /* type */ InputType::PT_TENSOR,
                  /* is_optional */ false,
                  /* allow_only_none */ std::nullopt,
                  /* allow_none */ std::nullopt,
                  /* ranks */ std::vector<int64_t>{1},
                  /* match_rank */ false,
                  /* dtypes */
                  std::vector<c10::ScalarType>{c10::ScalarType::Int},
                  /* match_precision_type */ false,
                  /* values */ std::nullopt,
                  /* is_array */ false,
                  /* array_length */ std::nullopt},
              InputDescriptor{
                  /* name */ "blank",
                  /* type */ InputType::NATIVE_INT,
                  /* is_optional */ false,
                  /* allow_only_none */ std::nullopt,
                  /* allow_none */ std::nullopt,
                  /* ranks */ std::nullopt,
                  /* match_rank */ std::nullopt,
                  /* dtypes */ std::nullopt,
                  /* match_precision_type */ std::nullopt,
                  /* values */ std::vector<std::any>{0},
                  /* is_array */ false,
                  /* array_length */ std::nullopt},
              InputDescriptor{
                  /* name */ "zero_infinity",
                  /* type */ InputType::NATIVE_BOOL,
                  /* is_optional */ false,
                  /* allow_only_none */ std::nullopt,
                  /* allow_none */ std::nullopt,
                  /* ranks */ std::nullopt,
                  /* match_rank */ std::nullopt,
                  /* dtypes */ std::nullopt,
                  /* match_precision_type */ std::nullopt,
                  /* values */ std::vector<std::any>{false},
                  /* is_array */ false,
                  /* array_length */ std::nullopt},
          },
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "_ctc_loss",
          "_ctc_loss.Tensor",
          {2})));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator__ctc_loss_Tensor));

  register_op(
      {/* op_name */ "_ctc_loss",
       /* overload */ "Tensor",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_index_reduce__exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, int dim, Tensor index, Tensor source, str reduce, bool include_self",
      "index_reduce_"));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_index_reduce));

  register_op(
      {/* op_name */ "index_reduce_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_in_place_interleave_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      /* schema */ "Tensor self",
      /* op_name */ "in_place_interleave",
      /* op_name_and_overload_name */ "",
      /* ranks */ std::vector<std::int64_t>{4},
      /* default_array_length */ 1,
      /* dim_size */ 4));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_in_place_interleave_));

  register_op(
      {/* op_name */ "in_place_interleave",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_kv_reorder_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "start",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{at::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "end",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{at::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "beam_idx",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{at::ScalarType::Byte},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "kv_reorder")));

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_kv_reorder_));

  register_op(
      {/* op_name */ "kv_reorder",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_linear_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor input, Tensor weight, Tensor? bias",
      "linear",
      "",
      std::vector<std::int64_t>{2}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_linear));

  register_op(
      {/* op_name */ "Linear",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "linear",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_fused_clip_norm_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      /* schema */ "Tensor[] grad, Tensor max_norm, float norm_type",
      /* op_name */ "fused_clip_norm",
      /* op_name_and_overload_name */ "",
      /* ranks*/ std::vector<int64_t>{1},
      /* default_array_length */ 2));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_fused_clip_norm));

  register_op(
      {/* op_name */ "fused_clip_norm",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_grid_sample_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners",
      "grid_sample"));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_grid_sampler_2d));

  register_op(
      {/* op_name */ "grid_sample",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());

  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_grid_sampler_3d));

  register_op(
      {/* op_name */ "grid_sample",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_im2col_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride",
      "im2col",
      "im2col",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_im2col));

  register_op(
      {/* op_name */ "im2col",
       /* overload */ "",
       /* op_namespace */ "torch.ops.aten"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_im2col_out_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, Tensor out",
      "im2col",
      "im2col.out",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_im2col_out));

  register_op(
      {/* op_name */ "im2col",
       /* overload */ "out",
       /* op_namespace */ "torch.ops.aten"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_masked_fill_exception() {
  /* SCALAR */
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "mask",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{c10::ScalarType::Char},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "value",
               /* type */ InputType::PT_SCALAR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "masked_fill",
          "masked_fill.Scalar")));

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_masked_fill_Scalar));

  register_op(
      {/* op_name */ "masked_fill",
       /* overload */ "Scalar",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "masked_fill",
       /* overload */ "Scalar",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_masked_fill__Scalar));

  register_op(
      {/* op_name */ "masked_fill_",
       /* overload */ "Scalar",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());

  /* TENSOR */
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "mask",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{c10::ScalarType::Char},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "value",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "masked_fill",
          "masked_fill.Tensor")));

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_masked_fill_Tensor));

  register_op(
      {/* op_name */ "masked_fill",
       /* overload */ "Tensor",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "masked_fill",
       /* overload */ "Tensor",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_masked_fill__Tensor));

  register_op(
      {/* op_name */ "masked_fill_",
       /* overload */ "Tensor",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_masked_scatter_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "mask",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Char},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "source",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "masked_scatter")));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_masked_scatter));

  register_op(
      {/* op_name */ "masked_scatter",
       /* overload */ "",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "masked_scatter",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());

  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_masked_scatter_));

  register_op(
      {/* op_name */ "masked_scatter_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_max_pool2d_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding, int[2] dilation, bool ceil_mode",
      "max_pool2d"));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_max_pool2d_with_indices));

  register_op(
      {/* op_name */ "max_pool2d",
       /* overload */ "",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "MaxPool2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "max_pool2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_max_pool3d_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding, int[3] dilation, bool ceil_mode",
      "max_pool3d"));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_max_pool3d_with_indices));

  register_op(
      {/* op_name */ "max_pool3d",
       /* overload */ "",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "MaxPool3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "max_pool3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_mm_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, Tensor mat2", "mm", "", std::vector<std::int64_t>{2}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_mm));

  register_op(
      {/* op_name */ "mm", /* overload */ "", /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "mm",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_mm_out_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, Tensor mat2, Tensor out",
      "mm",
      "mm.out",
      std::vector<std::int64_t>{2}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_mm_out));

  register_op(
      {/* op_name */ "mm", /* overload */ "out", /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_multi_margin_loss_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "target",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{
                   c10::ScalarType::Int, c10::ScalarType::Long},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "p",
               /* type */ InputType::PT_SCALAR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "margin",
               /* type */ InputType::PT_SCALAR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "weight",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ true,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ true,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "reduction",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "multi_margin_loss")));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_multi_margin_loss));

  register_op(
      {/* op_name */ "MultiMarginLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "multi_margin_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_multi_margin_loss_out_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "target",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{
                   c10::ScalarType::Int, c10::ScalarType::Long},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "p",
               /* type */ InputType::PT_SCALAR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "margin",
               /* type */ InputType::PT_SCALAR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "weight",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ true,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ true,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "reduction",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "out",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "multi_margin_loss",
          "multi_margin_loss.out")));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_multi_margin_loss_out));

  register_op(
      {/* op_name */ "multi_margin_loss",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_multilabel_margin_loss_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "target",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{
                   c10::ScalarType::Int, c10::ScalarType::Long},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "reduction",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "multilabel_margin_loss")));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_multilabel_margin_loss_forward));

  register_op(
      {/* op_name */ "MultiLabelMarginLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "multilabel_margin_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_nll_loss_forward_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "target",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "weight",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ true,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ true,
               /* ranks */ std::vector<int64_t>{1},
               /* match_rank */ false,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "reduction",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "ignore_index",
               /* type */ InputType::SYM_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{c10::SymInt(-100)},
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "nll_loss_forward",
          "",
          {2})));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_nll_loss_forward));

  register_op(
      {/* op_name */ "NLLLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "nll_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_nll_loss2d_forward));

  register_op(
      {/* op_name */ "NLLLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "nll_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_nll_loss_forward_output_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "target",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "weight",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ true,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ true,
               /* ranks */ std::vector<int64_t>{1},
               /* match_rank */ false,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "reduction",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "ignore_index",
               /* type */ InputType::SYM_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{c10::SymInt(-100)},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "output",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "total_weight",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::vector<int64_t>{1},
               /* match_rank */ false,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "nll_loss_forward",
          "nll_loss_forward.output",
          std::vector<std::int64_t>{2})));

  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_nll_loss_forward_output));

  register_op(
      {/* op_name */ "nll_loss",
       /* overload */ "output",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_nll_loss2d_forward_output));

  register_op(
      {/* op_name */ "nll_loss",
       /* overload */ "output",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::
    register_optimizer_resource_apply_momentum_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "params_momentum_buf_list",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ true,
               /* array_length */ 2},
           InputDescriptor{
               /* name */ "dp_list",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ true,
               /* array_length */ 1},
           InputDescriptor{
               /* name */ "momentum",
               /* type */ InputType::NATIVE_FLOAT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1.0F},
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "optimizer_resource_apply_momentum",
          "optimizer_resource_apply_momentum")));

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_optimizer_resource_apply_momentum));

  register_op(
      {/* op_name */ "optimizer_resource_apply_momentum",
       /* overload */ "",
       /* op_namespace */ "torch.hpu.optimizer"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_reflection_pad_exception() {
  /* REFLECTION_PAD_1D */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[2] padding",
      "reflection_pad1d",
      "",
      std::vector<std::int64_t>{3}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_reflection_pad1d));

  register_op(
      {/* op_name */ "ReflectionPad1d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());

  /* REFLECTION_PAD_2D */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[4] padding",
      "reflection_pad2d",
      "",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_reflection_pad2d));

  register_op(
      {/* op_name */ "ReflectionPad2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());

  /* REFLECTION_PAD_3D */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[6] padding",
      "reflection_pad3d",
      "",
      std::vector<std::int64_t>{5}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_reflection_pad3d));

  register_op(
      {/* op_name */ "ReflectionPad3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_replication_pad_exception() {
  /* REPLICATION_PAD_1D */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[2] padding",
      "replication_pad1d",
      "",
      std::vector<std::int64_t>{3}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_replication_pad1d));

  register_op(
      {/* op_name */ "ReplicationPad1d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());

  /* REPLICATION_PAD_2D */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[4] padding",
      "replication_pad2d",
      "",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_replication_pad2d));

  register_op(
      {/* op_name */ "ReplicationPad2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());

  /* REPLICATION_PAD_3D */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[6] padding",
      "replication_pad3d",
      "",
      std::vector<std::int64_t>{5}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_replication_pad3d));

  register_op(
      {/* op_name */ "ReplicationPad3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_rotary_pos_embedding_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "input",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "sin",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "cos",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "position_ids",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ true,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ true,
               /* ranks */ std::vector<int64_t>{2},
               /* match_rank */ false,
               /* dtypes */ std::vector<c10::ScalarType>{c10::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "offset",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{0},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "mode",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{0},
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "rotary_pos_embedding")));

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_rotary_pos_embedding));

  register_op(
      {/* op_name */ "rotary_pos_embedding",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      custom_executors.back().get());

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_rotary_pos_embedding_backward));

  register_op(
      {/* op_name */ "rotary_pos_embedding_backward",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::
    register_scaled_triangular_softmax_retain_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, float inv_scale_attn",
      "scaled_triangular_softmax_retain",
      "",
      std::vector<std::int64_t>{3}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_scaled_triangular_softmax_retain));

  register_op(
      {/* op_name */ "scaled_triangular_softmax_retain",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_scatter_add__exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, int dim, Tensor index, Tensor src", "scatter_add_"));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_scatter_add));

  register_op(
      {/* op_name */ "scatter_add_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_scatter_exception() {
  /* SRC */
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "dim",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "index",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{c10::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "src",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "scatter",
          "scatter.src",
          {3})));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_scatter_src));

  register_op(
      {/* op_name */ "scatter",
       /* overload */ "src",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "scatter",
       /* overload */ "src",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());

  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_scatter__src));

  register_op(
      {/* op_name */ "scatter_",
       /* overload */ "src",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());

  /* VALUE */
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "dim",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "index",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{c10::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "value",
               /* type */ InputType::PT_SCALAR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "scatter",
          "scatter.value",
          {3})));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_scatter_value));

  register_op(
      {/* op_name */ "scatter",
       /* overload */ "value",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "scatter",
       /* overload */ "value",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_scatter__value));

  register_op(
      {/* op_name */ "scatter_",
       /* overload */ "value",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_scatter_out_exception() {
  /* SRC */
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "dim",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "index",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{c10::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "src",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "out",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "scatter",
          "scatter.src_out",
          {3})));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_scatter_src_out));

  register_op(
      {/* op_name */ "scatter",
       /* overload */ "src_out",
       /* op_namespace */ "torch"},
      custom_executors.back().get());

  /* VALUE */
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "dim",
               /* type */ InputType::NATIVE_INT,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "index",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{c10::ScalarType::Int},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "value",
               /* type */ InputType::PT_SCALAR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ std::nullopt,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ std::nullopt,
               /* values */ std::vector<std::any>{1},
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "out",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "scatter",
          "scatter.value_out",
          {3})));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_scatter_value_out));

  register_op(
      {/* op_name */ "scatter",
       /* overload */ "value_out",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_searchsorted_exception() {
  /* TENSOR */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor sorted_sequence, Tensor self, bool out_int32, bool right, str? side, Tensor? sorter",
      "searchsorted",
      "searchsorted.Tensor",
      std::vector<std::int64_t>{2}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_searchsorted_Tensor));

  register_op(
      {/* op_name */ "searchsorted",
       /* overload */ "Tensor",
       /* op_namespace */ "torch"},
      custom_executors.back().get());

  /* TENSOR_OUT */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor sorted_sequence, Tensor self, bool out_int32, bool right, str? side, Tensor? sorter, Tensor out",
      "searchsorted",
      "searchsorted.Tensor_out",
      std::vector<std::int64_t>{2}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_searchsorted_Tensor_out));

  register_op(
      {/* op_name */ "searchsorted",
       /* overload */ "Tensor_out",
       /* op_namespace */ "torch"},
      custom_executors.back().get());

  /* SCALAR */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor sorted_sequence, Scalar self, bool out_int32, bool right, str? side, Tensor? sorter",
      "searchsorted",
      "searchsorted.Scalar"));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_searchsorted_Scalar));

  register_op(
      {/* op_name */ "searchsorted",
       /* overload */ "Scalar",
       /* op_namespace */ "torch"},
      custom_executors.back().get());

  /* SCALAR_OUT */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor sorted_sequence, Scalar self, bool out_int32, bool right, str? side, Tensor? sorter, Tensor out",
      "searchsorted",
      "searchsorted.Scalar_out"));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_searchsorted_Scalar_out));

  register_op(
      {/* op_name */ "searchsorted",
       /* overload */ "Scalar_out",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_max_unpool2d_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, Tensor indices, SymInt[2] output_size",
      "max_unpool2d",
      "max_unpool2d",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_max_unpool2d));

  register_op(
      {/* op_name */ "MaxUnpool2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "max_unpool2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_max_unpool2d_out_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, Tensor indices, SymInt[2] output_size, Tensor out",
      "max_unpool2d",
      "max_unpool2d.out",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_max_unpool2d_out));

  register_op(
      {/* op_name */ "max_unpool2d",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_max_unpool3d_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding",
      "max_unpool3d",
      "max_unpool3d",
      std::vector<std::int64_t>{5}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_max_unpool3d));

  register_op(
      {/* op_name */ "MaxUnpool3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "max_unpool3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_max_unpool3d_out_exception() {
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding, Tensor out",
      "max_unpool3d",
      "max_unpool3d.out",
      std::vector<std::int64_t>{5}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_max_unpool3d_out));

  register_op(
      {/* op_name */ "max_unpool3d",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_upsample_exception() {
  /* BILINEAR */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h, float? scales_w",
      "upsample_bilinear2d",
      "",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_upsample_bilinear2d));

  register_op(
      {/* op_name */ "Upsample",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "upsample",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator__upsample_bilinear2d_aa));

  register_op(
      {/* op_name */ "upsample_bilinear",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());

  /* BICUBIC */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h, float? scales_w",
      "upsample_bicubic2d",
      "",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_upsample_bilinear2d));

  register_op(
      {/* op_name */ "Upsample",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "upsample",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_upsample_out_exception() {
  /* BILINEAR */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h, float? scales_w, Tensor out",
      "upsample_bilinear2d",
      "upsample_bilinear2d.out",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_upsample_bilinear2d_out));

  register_op(
      {/* op_name */ "upsample",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator__upsample_bilinear2d_aa_out));

  register_op(
      {/* op_name */ "upsample_bilinear",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());

  /* BICUBIC */
  custom_stack_generators.push_back(std::make_unique<SchemaStackGenerator>(
      "Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h, float? scales_w, Tensor out",
      "upsample_bicubic2d",
      "upsample_bicubic2d.out",
      std::vector<std::int64_t>{4}));
  custom_executors.push_back(std::make_unique<GenericSharedLayerExecutor<>>(
      custom_stack_generators.back().get(),
      &habana::validator_upsample_bilinear2d_out));

  register_op(
      {/* op_name */ "upsample",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_where_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "condition",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{
                   c10::ScalarType::Float, c10::ScalarType::Char},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "other",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "where",
          "where.self")));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_where_self));

  register_op(
      {/* op_name */ "where",
       /* overload */ "self",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
  register_op(
      {/* op_name */ "where",
       /* overload */ "self",
       /* op_namespace */ "torch.Tensor"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_where_out_exception() {
  custom_stack_generators.push_back(
      std::make_unique<StackGenerator>(StackGenerator(
          {InputDescriptor{
               /* name */ "condition",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */
               std::vector<c10::ScalarType>{
                   c10::ScalarType::Float, c10::ScalarType::Char},
               /* match_precision_type */ false,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "self",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "other",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt},
           InputDescriptor{
               /* name */ "out",
               /* type */ InputType::PT_TENSOR,
               /* is_optional */ false,
               /* allow_only_none */ std::nullopt,
               /* allow_none */ std::nullopt,
               /* ranks */ std::nullopt,
               /* match_rank */ true,
               /* dtypes */ std::nullopt,
               /* match_precision_type */ true,
               /* values */ std::nullopt,
               /* is_array */ false,
               /* array_length */ std::nullopt}},
          /* blacklisted_precision_types */ {},
          /* whitelisted_precision_types */ {},
          "where",
          "where.self_out")));
  custom_executors.push_back(std::make_unique<CustomSharedLayerExecutor<>>(
      custom_stack_generators.back().get(), &habana::validator_where_self_out));

  register_op(
      {/* op_name */ "where",
       /* overload */ "self_out",
       /* op_namespace */ "torch"},
      custom_executors.back().get());
}

void SharedLayerReportGenerator::register_static_exceptions() {
  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Half}));
  auto fpExceptFp8Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Float8_e4m3fn,
          c10::ScalarType::Float8_e5m2}));
  auto fpExceptFp16Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{c10::ScalarType::Float}));
  auto fp32Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float, c10::ScalarType::BFloat16}));
  auto fp32Bf16Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Int}));
  auto fp32Bf16I32Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float, c10::ScalarType::Int}));
  auto fp32I32Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Int,
          c10::ScalarType::Bool}));
  auto fp32Bf16I32BoolExecutor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Int,
          c10::ScalarType::Char,
          c10::ScalarType::Bool}));
  auto fp32Bf16I32I8BoolExecutor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Int,
          c10::ScalarType::Char}));
  auto fp32Bf16I32I8Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Float8_e4m3fn,
          c10::ScalarType::Float8_e5m2,
          c10::ScalarType::Int,
          c10::ScalarType::Short}));
  auto i32I16AndFpExceptFp16Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Half,
          c10::ScalarType::Int}));
  auto i32AndFpExceptFp8Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Int,
          c10::ScalarType::Short}));
  auto i32I16AndF32BF16Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Half,
          c10::ScalarType::Float8_e4m3fn,
          c10::ScalarType::Float8_e5m2,
          c10::ScalarType::Long,
          c10::ScalarType::Int,
          c10::ScalarType::Short,
          c10::ScalarType::Char,
          c10::ScalarType::Bool}));
  auto allExecutor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Half,
          c10::ScalarType::Float8_e4m3fn,
          c10::ScalarType::Float8_e5m2,
          c10::ScalarType::Int,
          c10::ScalarType::Short,
          c10::ScalarType::Char,
          c10::ScalarType::Bool}));
  auto allExceptLongExecutor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Half,
          c10::ScalarType::Float8_e4m3fn,
          c10::ScalarType::Float8_e5m2,
          c10::ScalarType::Long,
          c10::ScalarType::Int,
          c10::ScalarType::Short}));
  auto allExceptI8AndBoolExecutor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Half,
          c10::ScalarType::Long,
          c10::ScalarType::Int,
          c10::ScalarType::Char,
          c10::ScalarType::Bool}));
  auto allExceptFp8I16Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Long, c10::ScalarType::Int}));
  auto i64I32Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Int, c10::ScalarType::Char, c10::ScalarType::Bool}));
  auto i32I8BoolExecutor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Long,
          c10::ScalarType::Int,
          c10::ScalarType::Short,
          c10::ScalarType::Char}));
  auto allIntegersExecutor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Float8_e4m3fn,
          c10::ScalarType::Float8_e5m2,
          c10::ScalarType::Int,
          c10::ScalarType::Short,
          c10::ScalarType::Char,
          c10::ScalarType::Bool}));
  auto allExceptFp16I64Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      std::vector<c10::ScalarType>{
          c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Float8_e4m3fn,
          c10::ScalarType::Float8_e5m2,
          c10::ScalarType::Int,
          c10::ScalarType::Char,
          c10::ScalarType::Bool}));
  auto allExceptFp16I64I16Executor = custom_executors.back().get();

  custom_executors.push_back(std::make_unique<StaticSharedLayerExecutor<>>(
      Report(/* fp4 */ false, /* int4 */ true)));
  auto i4Executor = custom_executors.back().get();

  /* __AND__ */
  register_op(
      {/* op_name */ "__and__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor);

  /* ACCUMULATE_GRADS_ */
  register_op(
      {/* op_name */ "accumulate_grads_",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      allExecutor);

  /* AS_STRIDED */
  register_op(
      {/* op_name */ "as_strided",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);

  register_op(
      {/* op_name */ "as_strided",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExecutor);

  register_op(
      {/* op_name */ "as_strided_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);

  register_op(
      {/* op_name */ "as_strided_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExecutor);

  /* BATCHED_NMS */
  register_op(
      {/* op_name */ "batched_nms",
       /* overload */ "",
       /* op_namespace */ "torchvision.ops"},
      fp32Bf16Executor);

  /* BATCH_NORM */
  register_op(
      {/* op_name */ "batch_norm",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "BatchNorm1d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "BatchNorm2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "batch_norm",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor);

  /* BINCOUNT */
  register_op(
      {/* op_name */ "bincount",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allIntegersExecutor);
  register_op(
      {/* op_name */ "bincount",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allIntegersExecutor);

  /* BROADCAST_TENSORS */
  register_op(
      {/* op_name */ "broadcast_tensors",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor);

  /* CHOLESKY */
  register_op(
      {/* op_name */ "cholesky",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Executor);
  register_op(
      {/* op_name */ "cholesky",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Executor);

  /* CHUNK */
  register_op(
      {/* op_name */ "chunk",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "chunk",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* CLIP */
  register_op(
      {/* op_name */ "clip",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "clip",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "clip_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "clip_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* CONVERT_FROM_INT4 */
  register_op(
      {/* op_name */ "convert_from_int4",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      i4Executor);

  /* CONVERT_FROM_UINT4 */
  register_op(
      {/* op_name */ "convert_from_uint4",
       /* overload */ "",
       /* op_namespace */ "torch.hpu"},
      i4Executor);

  /* CONJ */
  register_op(
      {/* op_name */ "conj",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "conj",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* CONSTANT_PAD_1D */
  register_op(
      {/* op_name */ "ConstantPad1d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor);

  /* COPY */
  register_op(
      {/* op_name */ "copy_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptFp16I64I16Executor);

  /* _COPY_FROM */
  register_op(
      {/* op_name */ "_copy_from",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);

  /* _COPY_FROM_AND_RESIZE */
  register_op(
      {/* op_name */ "_copy_from_and_resize",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);

  /* CROSS_ENTROPY_LOSS */
  register_op(
      {/* op_name */ "CrossEntropyLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fp32Bf16Executor);

  /* DEFORM_CONV2D */
  register_op(
      {/* op_name */ "deform_conv2d",
       /* overload */ "",
       /* op_namespace */ "torchvision.ops"},
      fp32Executor);

  /* DIAG */
  register_op(
      {/* op_name */ "diag",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "diag",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fpExceptFp8Executor);

  /* DROPOUT*/
  register_op(
      {/* op_name */ "dropout", /* overload */ "", /* op_namespace */ "torch"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "Dropout",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "dropout",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor);

  /* EMBEDDING */
  register_op(
      {/* op_name */ "embedding_bag",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "EmbeddingBag",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "embedding_bag",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor);

  /* EMPTY */
  register_op(
      {/* op_name */ "empty",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);
  register_op(
      {/* op_name */ "empty_like",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);
  register_op(
      {/* op_name */ "empty_strided",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);

  /* EXPAND_AS */
  register_op(
      {/* op_name */ "expand_as",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptFp8I16Executor);

  /* EXPIT */
  register_op(
      {/* op_name */ "expit",
       /* overload */ "",
       /* op_namespace */ "torch.special"},
      fp32Bf16Executor);

  /* FLATTEN */
  register_op(
      {/* op_name */ "flatten",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "flatten",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* FULL */
  register_op(
      {/* op_name */ "full",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);
  register_op(
      {/* op_name */ "full_like",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);

  /* __IAND__ */
  register_op(
      {/* op_name */ "__iand__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor);

  /* INDEX_ADD */
  register_op(
      {/* op_name */ "index_add",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i32AndFpExceptFp8Executor);

  register_op(
      {/* op_name */ "index_add",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32AndFpExceptFp8Executor);

  register_op(
      {/* op_name */ "index_add",
       /* overload */ "out",
       /* op_namespace */ "torch"},
      i32AndFpExceptFp8Executor);

  register_op(
      {/* op_name */ "index_add_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32AndFpExceptFp8Executor);

  /* INDEX_PUT */
  register_op(
      {/* op_name */ "index_put",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "index_put_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "index_put",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "index_put_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* INSTANCE_NORM */
  register_op(
      {/* op_name */ "instance_norm",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "InstanceNorm2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "instance_norm",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fp32Bf16Executor);

  /* __IOR__ */
  register_op(
      {/* op_name */ "__ior__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor);

  /* IS_COMPLEX */
  register_op(
      {/* op_name */ "is_complex",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "is_complex",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* IS_FLOATING_POINT */
  register_op(
      {/* op_name */ "is_floating_point",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "is_floating_point",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* IS_NONZERO */
  register_op(
      {/* op_name */ "is_nonzero",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "is_nonzero",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* ITEM */
  register_op(
      {/* op_name */ "item",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Executor);

  /* __IXOR__ */
  register_op(
      {/* op_name */ "__ixor__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor);

  /* LAYER_NORM */
  register_op(
      {/* op_name */ "layer_norm",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "LayerNorm",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "layer_norm",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor);

  /* LOGSUMEXP */
  register_op(
      {/* op_name */ "logsumexp",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "logsumexp",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "logsumexp",
       /* overload */ "",
       /* op_namespace */ "torch.special"},
      fp32Bf16Executor);

  /* L1_LOSS */
  register_op(
      {/* op_name */ "l1_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fp32Bf16Executor);

  /* MASKED_SELECT */
  register_op(
      {/* op_name */ "masked_select",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "masked_select",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* MATMUL */
  register_op(
      {/* op_name */ "matmul",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "matmul",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16Executor);

  /* MESHGRID */
  register_op(
      {/* op_name */ "meshgrid",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor);

  /* NARROW */
  register_op(
      {/* op_name */ "narrow",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Executor);
  register_op(
      {/* op_name */ "narrow",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Executor);

  /* _NATIVE_BATCH_NORM_LEGIT */
  register_op(
      {/* op_name */ "_native_batch_norm_legit",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);

  /* _NATIVE_BATCH_NORM_LEGIT_NO_TRAINING */
  register_op(
      {/* op_name */ "_native_batch_norm_legit_no_training",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);

  /* NATIVE_LAYER_NORM */
  register_op(
      {/* op_name */ "native_layer_norm",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);

  /* NEW_EMPTY */
  register_op(
      {/* op_name */ "new_empty",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8Executor);

  /* NEW_EMPTY_STRIDED */
  register_op(
      {/* op_name */ "new_empty_strided",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8Executor);

  /* NEW_FULL */
  register_op(
      {/* op_name */ "new_full",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8Executor);

  /* NEW_ONES */
  register_op(
      {/* op_name */ "new_ones",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8Executor);

  /* NMS */
  register_op(
      {/* op_name */ "nms",
       /* overload */ "",
       /* op_namespace */ "torchvision.ops"},
      fp32Bf16Executor);

  /* NONZERO */
  register_op(
      {/* op_name */ "nonzero",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32BoolExecutor);
  register_op(
      {/* op_name */ "nonzero",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32BoolExecutor);

  /* ONES */
  register_op(
      {/* op_name */ "ones",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptFp16I64I16Executor);

  /* ONES_LIKE */
  register_op(
      {/* op_name */ "ones_like",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptFp16I64Executor);

  /* __OR__ */
  register_op(
      {/* op_name */ "__or__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor);

  /* PAD */
  register_op(
      {/* op_name */ "pad",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp16Executor);

  /* PRELU */
  register_op(
      {/* op_name */ "prelu",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "PReLU",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "prelu",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fp32Bf16Executor);

  /* REPEAT_INTERLEAVE */
  register_op(
      {/* op_name */ "repeat_interleave",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptFp8I16Executor);

  /* RESHAPE */
  register_op(
      {/* op_name */ "reshape",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor);
  register_op(
      {/* op_name */ "reshape",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExecutor);

  /* RESOLVE_CONJ */
  register_op(
      {/* op_name */ "resolve_conj",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "resolve_conj",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16Executor);

  /* RESOLVE_NEG */
  register_op(
      {/* op_name */ "resolve_neg",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor);
  register_op(
      {/* op_name */ "resolve_neg",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16Executor);

  /* RESULT_TYPE */
  register_op(
      {/* op_name */ "result_type",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);

  /* RESULT_TYPE */
  register_op(
      {/* op_name */ "pin_memory",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8BoolExecutor);

  /* ROI_ALIGN */
  register_op(
      {/* op_name */ "roi_align",
       /* overload */ "",
       /* op_namespace */ "torchvision.ops"},
      fp32Executor);

  /* SPLIT_WITH_SIZES */
  register_op(
      {/* op_name */ "split_with_sizes",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i32I16AndFpExceptFp16Executor);
  register_op(
      {/* op_name */ "split_with_sizes",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I16AndFpExceptFp16Executor);

  /* SQUARE */
  register_op(
      {/* op_name */ "square",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32I8BoolExecutor);
  register_op(
      {/* op_name */ "square",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8BoolExecutor);
  register_op(
      {/* op_name */ "square_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32I8BoolExecutor);
  register_op(
      {/* op_name */ "square_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8BoolExecutor);

  /* SOFTMAX */
  register_op(
      {/* op_name */ "softmax",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "Softmax",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "softmax",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor);
  register_op(
      {/* op_name */ "softmax",
       /* overload */ "",
       /* op_namespace */ "torch.special"},
      fpExceptFp8Executor);

  /* STACK */
  register_op(
      {/* op_name */ "stack",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i32I16AndF32BF16Executor);

  /* T */
  register_op(
      {/* op_name */ "T",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExecutor);

  /* to */
  register_op(
      {/* op_name */ "to",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExecutor);

  /* TRIL_INDICES */
  register_op(
      {/* op_name */ "tril_indices",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i64I32Executor);

  /* TRIU_INDICES */
  register_op(
      {/* op_name */ "triu_indices",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i64I32Executor);

  /* UNBIND */
  register_op(
      {/* op_name */ "unbind",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor);
  register_op(
      {/* op_name */ "unbind",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor);

  /* UNIQUE */
  register_op(
      {/* op_name */ "unique",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32I32Executor);
  register_op(
      {/* op_name */ "unique",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32I32Executor);

  /* _UNIQUE */
  register_op(
      {/* op_name */ "_unique",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32I32Executor);

  /* _UNIQUE2 */
  register_op(
      {/* op_name */ "_unique2",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32I32Executor);

  /* UNSQUEEZE */
  register_op(
      {/* op_name */ "unsqueeze",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptI8AndBoolExecutor);
  register_op(
      {/* op_name */ "unsqueeze",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptI8AndBoolExecutor);
  register_op(
      {/* op_name */ "unsqueeze_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptI8AndBoolExecutor);

  /* WEIGHT_NORM */
  register_op(
      {/* op_name */ "weight_norm",
       /* overload */ "",
       /* op_namespace */ "torch.nn.utils"},
      fp32Bf16Executor);

  /* _WEIGHT_NORM_INTERFACE */
  register_op(
      {/* op_name */ "_weight_norm_interface",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor);

  /* __XOR__ */
  register_op(
      {/* op_name */ "__xor__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor);

  /* ZEROS */
  register_op(
      {/* op_name */ "zeros",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptLongExecutor);

  /* ZEROS_LIKE */
  register_op(
      {/* op_name */ "zeros_like",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptLongExecutor);
}

} // namespace slrg
