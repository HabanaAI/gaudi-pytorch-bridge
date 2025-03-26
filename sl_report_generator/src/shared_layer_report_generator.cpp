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
  register_ctc_loss_exception();
  register_ctc_loss_tensor_exception();
  register_grid_sample_exception();
  register_im2col_exception();
  register_im2col_out_exception();
  register_index_reduce__exception();
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
  register_reflection_pad_exception();
  register_replication_pad_exception();
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
  std::shared_ptr<IStackGenerator> adaptiveAvgPool2dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[2] output_size", "_adaptive_avg_pool2d");
  std::shared_ptr<SharedLayerExecutor<>> adaptiveAvgPool2dExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          adaptiveAvgPool2dStackGenerator.get(),
          &habana::validator__adaptive_avg_pool2d);
  custom_stack_generators.push_back(adaptiveAvgPool2dStackGenerator);
  custom_executors.push_back(adaptiveAvgPool2dExecutor);
  register_op(
      {/* op_name */ "_adaptive_avg_pool2d",
       /* overload */ "",
       /* op_namespace */ "torch"},
      adaptiveAvgPool2dExecutor.get());
}

void SharedLayerReportGenerator::register_abs__exception() {
  std::shared_ptr<IStackGenerator> absStackGenerator =
      std::make_shared<SchemaStackGenerator>("Tensor self", "abs_");
  std::shared_ptr<SharedLayerExecutor<>> absExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          absStackGenerator.get(), &habana::validator_abs_);
  custom_stack_generators.push_back(absStackGenerator);
  custom_executors.push_back(absExecutor);
  register_op(
      {/* op_name */ "abs_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      absExecutor.get());
}

void SharedLayerReportGenerator::register_bmm_exception() {
  std::shared_ptr<IStackGenerator> bmmStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, Tensor mat2", "bmm", "", std::vector<std::int64_t>{3});
  std::shared_ptr<SharedLayerExecutor<>> bmmExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          bmmStackGenerator.get(), &habana::validator_bmm);
  custom_stack_generators.push_back(bmmStackGenerator);
  custom_executors.push_back(bmmExecutor);
  register_op(
      {/* op_name */ "bmm", /* overload */ "", /* op_namespace */ "torch"},
      bmmExecutor.get());
  register_op(
      {/* op_name */ "bmm",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      bmmExecutor.get());
}

void SharedLayerReportGenerator::register_bmm_out_exception() {
  std::shared_ptr<IStackGenerator> bmmOutStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, Tensor mat2, Tensor out",
          "bmm",
          "bmm.out",
          std::vector<std::int64_t>{3});
  std::shared_ptr<SharedLayerExecutor<>> bmmOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          bmmOutStackGenerator.get(), &habana::validator_bmm_out);
  custom_stack_generators.push_back(bmmOutStackGenerator);
  custom_executors.push_back(bmmOutExecutor);
  register_op(
      {/* op_name */ "bmm", /* overload */ "out", /* op_namespace */ "torch"},
      bmmOutExecutor.get());
}

void SharedLayerReportGenerator::register_channel_shuffle_exception() {
  std::shared_ptr<IStackGenerator> channelShuffleStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt groups",
          "channel_shuffle",
          "",
          std::vector<std::int64_t>{3});
  std::shared_ptr<SharedLayerExecutor<>> channelShuffleExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          channelShuffleStackGenerator.get(),
          &habana::validator_channel_shuffle);
  custom_stack_generators.push_back(channelShuffleStackGenerator);
  custom_executors.push_back(channelShuffleExecutor);
  register_op(
      {/* op_name */ "channel_shuffle",
       /* overload */ "",
       /* op_namespace */ "torch"},
      channelShuffleExecutor.get());
  register_op(
      {/* op_name */ "ChannelShuffle",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      channelShuffleExecutor.get());
  register_op(
      {/* op_name */ "channel_shuffle",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      channelShuffleExecutor.get());
}

void SharedLayerReportGenerator::register_clamp_exception() {
  std::shared_ptr<IStackGenerator> clampStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "clamp"));
  std::shared_ptr<SharedLayerExecutor<>> clampExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          clampStackGenerator.get(), &habana::validator_clamp);
  custom_stack_generators.push_back(clampStackGenerator);
  custom_executors.push_back(clampExecutor);
  register_op(
      {/* op_name */ "clamp", /* overload */ "", /* op_namespace */ "torch"},
      clampExecutor.get());
  register_op(
      {/* op_name */ "clamp",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      clampExecutor.get());
}

void SharedLayerReportGenerator::register_ctc_loss_exception() {
  std::shared_ptr<IStackGenerator> ctcLossStackGenerator = std::make_shared<
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
      {2}));
  std::shared_ptr<SharedLayerExecutor<>> ctcLossExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          ctcLossStackGenerator.get(), &habana::validator__ctc_loss);
  custom_stack_generators.push_back(ctcLossStackGenerator);
  custom_executors.push_back(ctcLossExecutor);
  register_op(
      {/* op_name */ "_ctc_loss",
       /* overload */ "",
       /* op_namespace */ "torch"},
      ctcLossExecutor.get());
}

void SharedLayerReportGenerator::register_ctc_loss_tensor_exception() {
  std::shared_ptr<IStackGenerator> ctcLossTensorStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          {2}));
  std::shared_ptr<SharedLayerExecutor<>> ctcLossTensorExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          ctcLossTensorStackGenerator.get(),
          &habana::validator__ctc_loss_Tensor);
  custom_stack_generators.push_back(ctcLossTensorStackGenerator);
  custom_executors.push_back(ctcLossTensorExecutor);
  register_op(
      {/* op_name */ "_ctc_loss",
       /* overload */ "Tensor",
       /* op_namespace */ "torch"},
      ctcLossTensorExecutor.get());
}

void SharedLayerReportGenerator::register_index_reduce__exception() {
  std::shared_ptr<IStackGenerator> indexReduceStackGenerator = std::make_shared<
      SchemaStackGenerator>(
      "Tensor self, int dim, Tensor index, Tensor source, str reduce, bool include_self",
      "index_reduce_");
  std::shared_ptr<SharedLayerExecutor<>> indexReduceExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          indexReduceStackGenerator.get(), &habana::validator_index_reduce);
  custom_stack_generators.push_back(indexReduceStackGenerator);
  custom_executors.push_back(indexReduceExecutor);
  register_op(
      {/* op_name */ "index_reduce_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      indexReduceExecutor.get());
}

void SharedLayerReportGenerator::register_linear_exception() {
  std::shared_ptr<IStackGenerator> linearStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor input, Tensor weight, Tensor? bias",
          "linear",
          "",
          std::vector<std::int64_t>{2});
  std::shared_ptr<SharedLayerExecutor<>> linearExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          linearStackGenerator.get(), &habana::validator_linear);
  custom_stack_generators.push_back(linearStackGenerator);
  custom_executors.push_back(linearExecutor);
  register_op(
      {/* op_name */ "Linear",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      linearExecutor.get());
  register_op(
      {/* op_name */ "linear",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      linearExecutor.get());
}

void SharedLayerReportGenerator::register_grid_sample_exception() {
  std::shared_ptr<IStackGenerator> gridSampleStackGenerator = std::make_shared<
      SchemaStackGenerator>(
      "Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners",
      "grid_sample");
  std::shared_ptr<SharedLayerExecutor<>> gridSampler2dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          gridSampleStackGenerator.get(), &habana::validator_grid_sampler_2d);
  std::shared_ptr<SharedLayerExecutor<>> gridSampler3dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          gridSampleStackGenerator.get(), &habana::validator_grid_sampler_3d);
  custom_stack_generators.push_back(gridSampleStackGenerator);
  custom_executors.push_back(gridSampler2dExecutor);
  custom_executors.push_back(gridSampler3dExecutor);
  register_op(
      {/* op_name */ "grid_sample",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      gridSampler2dExecutor.get());
  register_op(
      {/* op_name */ "grid_sample",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      gridSampler3dExecutor.get());
}

void SharedLayerReportGenerator::register_im2col_exception() {
  std::shared_ptr<IStackGenerator> im2colStackGenerator = std::make_shared<
      SchemaStackGenerator>(
      "Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride",
      "im2col",
      "im2col",
      std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> im2colExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          im2colStackGenerator.get(), &habana::validator_im2col);
  custom_stack_generators.push_back(im2colStackGenerator);
  custom_executors.push_back(im2colExecutor);
  register_op(
      {/* op_name */ "im2col",
       /* overload */ "",
       /* op_namespace */ "torch.ops.aten"},
      im2colExecutor.get());
}

void SharedLayerReportGenerator::register_im2col_out_exception() {
  std::shared_ptr<IStackGenerator> im2colStackGenerator = std::make_shared<
      SchemaStackGenerator>(
      "Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, Tensor out",
      "im2col",
      "im2col.out",
      std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> im2colExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          im2colStackGenerator.get(), &habana::validator_im2col_out);
  custom_stack_generators.push_back(im2colStackGenerator);
  custom_executors.push_back(im2colExecutor);
  register_op(
      {/* op_name */ "im2col",
       /* overload */ "out",
       /* op_namespace */ "torch.ops.aten"},
      im2colExecutor.get());
}

void SharedLayerReportGenerator::register_masked_fill_exception() {
  /* SCALAR */
  std::shared_ptr<IStackGenerator> maskedFillScalarStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "masked_fill.Scalar"));
  std::shared_ptr<SharedLayerExecutor<>> maskedFillScalarExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          maskedFillScalarStackGenerator.get(),
          &habana::validator_masked_fill_Scalar);
  std::shared_ptr<SharedLayerExecutor<>> maskedFillScalarInplaceExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          maskedFillScalarStackGenerator.get(),
          &habana::validator_masked_fill__Scalar);
  custom_stack_generators.push_back(maskedFillScalarStackGenerator);
  custom_executors.push_back(maskedFillScalarExecutor);
  custom_executors.push_back(maskedFillScalarInplaceExecutor);
  register_op(
      {/* op_name */ "masked_fill",
       /* overload */ "Scalar",
       /* op_namespace */ "torch"},
      maskedFillScalarExecutor.get());
  register_op(
      {/* op_name */ "masked_fill",
       /* overload */ "Scalar",
       /* op_namespace */ "torch.Tensor"},
      maskedFillScalarExecutor.get());
  register_op(
      {/* op_name */ "masked_fill_",
       /* overload */ "Scalar",
       /* op_namespace */ "torch.Tensor"},
      maskedFillScalarInplaceExecutor.get());

  /* TENSOR */
  std::shared_ptr<IStackGenerator> maskedFillTensorStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "masked_fill.Tensor"));
  std::shared_ptr<SharedLayerExecutor<>> maskedFillTensorExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          maskedFillTensorStackGenerator.get(),
          &habana::validator_masked_fill_Tensor);
  std::shared_ptr<SharedLayerExecutor<>> maskedFillTensorInplaceExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          maskedFillTensorStackGenerator.get(),
          &habana::validator_masked_fill__Tensor);
  custom_stack_generators.push_back(maskedFillTensorStackGenerator);
  custom_executors.push_back(maskedFillTensorExecutor);
  custom_executors.push_back(maskedFillTensorInplaceExecutor);
  register_op(
      {/* op_name */ "masked_fill",
       /* overload */ "Tensor",
       /* op_namespace */ "torch"},
      maskedFillTensorExecutor.get());
  register_op(
      {/* op_name */ "masked_fill",
       /* overload */ "Tensor",
       /* op_namespace */ "torch.Tensor"},
      maskedFillTensorExecutor.get());
  register_op(
      {/* op_name */ "masked_fill_",
       /* overload */ "Tensor",
       /* op_namespace */ "torch.Tensor"},
      maskedFillTensorInplaceExecutor.get());
}

/*Tensor self, Tensor mask, Tensor source
extern CheckNodeWithSharedLayerValidator validator_masked_scatter;
extern CheckNodeWithSharedLayerValidator validator_masked_scatter_;
*/
void SharedLayerReportGenerator::register_masked_scatter_exception() {
  std::shared_ptr<IStackGenerator> maskedScatterStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "masked_scatter"));
  std::shared_ptr<SharedLayerExecutor<>> maskedScatterExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          maskedScatterStackGenerator.get(), &habana::validator_masked_scatter);
  std::shared_ptr<SharedLayerExecutor<>> maskedScatterInplaceExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          maskedScatterStackGenerator.get(),
          &habana::validator_masked_scatter_);
  custom_stack_generators.push_back(maskedScatterStackGenerator);
  custom_executors.push_back(maskedScatterExecutor);
  custom_executors.push_back(maskedScatterInplaceExecutor);
  register_op(
      {/* op_name */ "masked_scatter",
       /* overload */ "",
       /* op_namespace */ "torch"},
      maskedScatterExecutor.get());
  register_op(
      {/* op_name */ "masked_scatter",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      maskedScatterExecutor.get());
  register_op(
      {/* op_name */ "masked_scatter_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      maskedScatterInplaceExecutor.get());
}

void SharedLayerReportGenerator::register_max_pool2d_exception() {
  std::shared_ptr<IStackGenerator> maxPool2dStackGenerator = std::make_shared<
      SchemaStackGenerator>(
      "Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding, int[2] dilation, bool ceil_mode",
      "max_pool2d");
  std::shared_ptr<SharedLayerExecutor<>> maxPool2dExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          maxPool2dStackGenerator.get(),
          &habana::validator_max_pool2d_with_indices);
  custom_stack_generators.push_back(maxPool2dStackGenerator);
  custom_executors.push_back(maxPool2dExecutor);
  register_op(
      {/* op_name */ "max_pool2d",
       /* overload */ "",
       /* op_namespace */ "torch"},
      maxPool2dExecutor.get());
  register_op(
      {/* op_name */ "MaxPool2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      maxPool2dExecutor.get());
  register_op(
      {/* op_name */ "max_pool2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      maxPool2dExecutor.get());
}

void SharedLayerReportGenerator::register_max_pool3d_exception() {
  std::shared_ptr<IStackGenerator> maxPool3dStackGenerator = std::make_shared<
      SchemaStackGenerator>(
      "Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding, int[3] dilation, bool ceil_mode",
      "max_pool3d");
  std::shared_ptr<SharedLayerExecutor<>> maxPool3dExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          maxPool3dStackGenerator.get(),
          &habana::validator_max_pool3d_with_indices);
  custom_stack_generators.push_back(maxPool3dStackGenerator);
  custom_executors.push_back(maxPool3dExecutor);
  register_op(
      {/* op_name */ "max_pool3d",
       /* overload */ "",
       /* op_namespace */ "torch"},
      maxPool3dExecutor.get());
  register_op(
      {/* op_name */ "MaxPool3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      maxPool3dExecutor.get());
  register_op(
      {/* op_name */ "max_pool3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      maxPool3dExecutor.get());
}

void SharedLayerReportGenerator::register_mm_exception() {
  std::shared_ptr<IStackGenerator> mmStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, Tensor mat2", "mm", "", std::vector<std::int64_t>{2});
  std::shared_ptr<SharedLayerExecutor<>> mmExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          mmStackGenerator.get(), &habana::validator_mm);
  custom_stack_generators.push_back(mmStackGenerator);
  custom_executors.push_back(mmExecutor);
  register_op(
      {/* op_name */ "mm", /* overload */ "", /* op_namespace */ "torch"},
      mmExecutor.get());
  register_op(
      {/* op_name */ "mm",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      mmExecutor.get());
}

void SharedLayerReportGenerator::register_mm_out_exception() {
  std::shared_ptr<IStackGenerator> mmOutStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, Tensor mat2, Tensor out",
          "mm",
          "mm.out",
          std::vector<std::int64_t>{2});
  std::shared_ptr<SharedLayerExecutor<>> mmOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          mmOutStackGenerator.get(), &habana::validator_mm_out);
  custom_stack_generators.push_back(mmOutStackGenerator);
  custom_executors.push_back(mmOutExecutor);
  register_op(
      {/* op_name */ "mm", /* overload */ "out", /* op_namespace */ "torch"},
      mmOutExecutor.get());
}

void SharedLayerReportGenerator::register_multi_margin_loss_exception() {
  std::shared_ptr<IStackGenerator> multiMarginLossStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "multi_margin_loss"));
  std::shared_ptr<SharedLayerExecutor<>> multiMarginLossExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          multiMarginLossStackGenerator.get(),
          &habana::validator_multi_margin_loss);
  custom_stack_generators.push_back(multiMarginLossStackGenerator);
  custom_executors.push_back(multiMarginLossExecutor);
  register_op(
      {/* op_name */ "MultiMarginLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      multiMarginLossExecutor.get());
  register_op(
      {/* op_name */ "multi_margin_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      multiMarginLossExecutor.get());
}

void SharedLayerReportGenerator::register_multi_margin_loss_out_exception() {
  std::shared_ptr<IStackGenerator> multiMarginLossOutStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "multi_margin_loss.out"));
  std::shared_ptr<SharedLayerExecutor<>> multiMarginLossOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          multiMarginLossOutStackGenerator.get(),
          &habana::validator_multi_margin_loss_out);
  custom_stack_generators.push_back(multiMarginLossOutStackGenerator);
  custom_executors.push_back(multiMarginLossOutExecutor);
  register_op(
      {/* op_name */ "multi_margin_loss",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      multiMarginLossOutExecutor.get());
}

void SharedLayerReportGenerator::register_multilabel_margin_loss_exception() {
  std::shared_ptr<IStackGenerator> multilabelMarginLossStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "multilabel_margin_loss"));
  std::shared_ptr<SharedLayerExecutor<>> multilabelMarginLossExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          multilabelMarginLossStackGenerator.get(),
          &habana::validator_multilabel_margin_loss_forward);
  custom_stack_generators.push_back(multilabelMarginLossStackGenerator);
  custom_executors.push_back(multilabelMarginLossExecutor);
  register_op(
      {/* op_name */ "MultiLabelMarginLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      multilabelMarginLossExecutor.get());
  register_op(
      {/* op_name */ "multilabel_margin_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      multilabelMarginLossExecutor.get());
}

void SharedLayerReportGenerator::register_nll_loss_forward_exception() {
  std::shared_ptr<IStackGenerator> nllLossForwardStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          {2}));
  std::shared_ptr<SharedLayerExecutor<>> nllLossForwardExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          nllLossForwardStackGenerator.get(),
          &habana::validator_nll_loss_forward);
  std::shared_ptr<SharedLayerExecutor<>> nllLoss2dForwardExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          nllLossForwardStackGenerator.get(),
          &habana::validator_nll_loss2d_forward);
  custom_stack_generators.push_back(nllLossForwardStackGenerator);
  custom_executors.push_back(nllLossForwardExecutor);
  custom_executors.push_back(nllLoss2dForwardExecutor);
  register_op(
      {/* op_name */ "NLLLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      nllLossForwardExecutor.get());
  register_op(
      {/* op_name */ "nll_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      nllLossForwardExecutor.get());
  register_op(
      {/* op_name */ "NLLLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      nllLoss2dForwardExecutor.get());
  register_op(
      {/* op_name */ "nll_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      nllLoss2dForwardExecutor.get());
}

void SharedLayerReportGenerator::register_nll_loss_forward_output_exception() {
  std::shared_ptr<IStackGenerator> nllLossForwardOutStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          std::vector<std::int64_t>{2}));
  std::shared_ptr<SharedLayerExecutor<>> nllLossForwardOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          nllLossForwardOutStackGenerator.get(),
          &habana::validator_nll_loss_forward_output);
  std::shared_ptr<SharedLayerExecutor<>> nllLoss2dForwardOutExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          nllLossForwardOutStackGenerator.get(),
          &habana::validator_nll_loss2d_forward_output);
  custom_stack_generators.push_back(nllLossForwardOutStackGenerator);
  custom_executors.push_back(nllLossForwardOutExecutor);
  custom_executors.push_back(nllLoss2dForwardOutExecutor);
  register_op(
      {/* op_name */ "nll_loss",
       /* overload */ "output",
       /* op_namespace */ "torch.nn.functional"},
      nllLossForwardOutExecutor.get());
  register_op(
      {/* op_name */ "nll_loss",
       /* overload */ "output",
       /* op_namespace */ "torch.nn.functional"},
      nllLoss2dForwardOutExecutor.get());
}

void SharedLayerReportGenerator::register_reflection_pad_exception() {
  /* REFLECTION_PAD_1D */
  std::shared_ptr<IStackGenerator> reflectionPad1dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[2] padding",
          "reflection_pad1d",
          "",
          std::vector<std::int64_t>{3});
  std::shared_ptr<SharedLayerExecutor<>> reflectionPad1dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          reflectionPad1dStackGenerator.get(),
          &habana::validator_reflection_pad1d);
  custom_stack_generators.push_back(reflectionPad1dStackGenerator);
  custom_executors.push_back(reflectionPad1dExecutor);

  register_op(
      {/* op_name */ "ReflectionPad1d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      reflectionPad1dExecutor.get());

  /* REFLECTION_PAD_2D */
  std::shared_ptr<IStackGenerator> reflectionPad2dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[4] padding",
          "reflection_pad2d",
          "",
          std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> reflectionPad2dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          reflectionPad2dStackGenerator.get(),
          &habana::validator_reflection_pad2d);
  custom_stack_generators.push_back(reflectionPad2dStackGenerator);
  custom_executors.push_back(reflectionPad2dExecutor);

  register_op(
      {/* op_name */ "ReflectionPad2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      reflectionPad2dExecutor.get());

  /* REFLECTION_PAD_3D */
  std::shared_ptr<IStackGenerator> reflectionPad3dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[6] padding",
          "reflection_pad3d",
          "",
          std::vector<std::int64_t>{5});
  std::shared_ptr<SharedLayerExecutor<>> reflectionPad3dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          reflectionPad3dStackGenerator.get(),
          &habana::validator_reflection_pad3d);
  custom_stack_generators.push_back(reflectionPad3dStackGenerator);
  custom_executors.push_back(reflectionPad3dExecutor);

  register_op(
      {/* op_name */ "ReflectionPad3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      reflectionPad3dExecutor.get());
}

void SharedLayerReportGenerator::register_replication_pad_exception() {
  /* REPLICATION_PAD_1D */
  std::shared_ptr<IStackGenerator> replicationPad1dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[2] padding",
          "replication_pad1d",
          "",
          std::vector<std::int64_t>{3});
  std::shared_ptr<SharedLayerExecutor<>> replicationPad1dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          replicationPad1dStackGenerator.get(),
          &habana::validator_replication_pad1d);
  custom_stack_generators.push_back(replicationPad1dStackGenerator);
  custom_executors.push_back(replicationPad1dExecutor);

  register_op(
      {/* op_name */ "ReplicationPad1d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      replicationPad1dExecutor.get());

  /* REPLICATION_PAD_2D */
  std::shared_ptr<IStackGenerator> replicationPad2dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[4] padding",
          "replication_pad2d",
          "",
          std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> replicationPad2dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          replicationPad2dStackGenerator.get(),
          &habana::validator_replication_pad2d);
  custom_stack_generators.push_back(replicationPad2dStackGenerator);
  custom_executors.push_back(replicationPad2dExecutor);

  register_op(
      {/* op_name */ "ReplicationPad2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      replicationPad2dExecutor.get());

  /* REPLICATION_PAD_3D */
  std::shared_ptr<IStackGenerator> replicationPad3dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[6] padding",
          "replication_pad3d",
          "",
          std::vector<std::int64_t>{5});
  std::shared_ptr<SharedLayerExecutor<>> replicationPad3dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          replicationPad3dStackGenerator.get(),
          &habana::validator_replication_pad3d);
  custom_stack_generators.push_back(replicationPad3dStackGenerator);
  custom_executors.push_back(replicationPad3dExecutor);

  register_op(
      {/* op_name */ "ReplicationPad3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      replicationPad3dExecutor.get());
}

void SharedLayerReportGenerator::register_scatter_add__exception() {
  std::shared_ptr<IStackGenerator> scatterAddStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, int dim, Tensor index, Tensor src", "scatter_add_");
  std::shared_ptr<SharedLayerExecutor<>> scatterAddExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          scatterAddStackGenerator.get(), &habana::validator_scatter_add);
  custom_stack_generators.push_back(scatterAddStackGenerator);
  custom_executors.push_back(scatterAddExecutor);

  register_op(
      {/* op_name */ "scatter_add_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      scatterAddExecutor.get());
}

void SharedLayerReportGenerator::register_scatter_exception() {
  /* SRC */
  std::shared_ptr<IStackGenerator> scatterSrcStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          {3}));
  std::shared_ptr<SharedLayerExecutor<>> scatterSrcExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          scatterSrcStackGenerator.get(), &habana::validator_scatter_src);
  std::shared_ptr<SharedLayerExecutor<>> scatterSrcInplaceExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          scatterSrcStackGenerator.get(), &habana::validator_scatter__src);
  custom_stack_generators.push_back(scatterSrcStackGenerator);
  custom_executors.push_back(scatterSrcExecutor);
  custom_executors.push_back(scatterSrcInplaceExecutor);
  register_op(
      {/* op_name */ "scatter",
       /* overload */ "src",
       /* op_namespace */ "torch"},
      scatterSrcExecutor.get());
  register_op(
      {/* op_name */ "scatter",
       /* overload */ "src",
       /* op_namespace */ "torch.Tensor"},
      scatterSrcExecutor.get());
  register_op(
      {/* op_name */ "scatter_",
       /* overload */ "src",
       /* op_namespace */ "torch.Tensor"},
      scatterSrcInplaceExecutor.get());

  /* VALUE */
  std::shared_ptr<IStackGenerator> scatterValueStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          {3}));
  std::shared_ptr<SharedLayerExecutor<>> scatterValueExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          scatterValueStackGenerator.get(), &habana::validator_scatter_value);
  std::shared_ptr<SharedLayerExecutor<>> scatterValueInplaceExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          scatterValueStackGenerator.get(), &habana::validator_scatter__value);
  custom_stack_generators.push_back(scatterValueStackGenerator);
  custom_executors.push_back(scatterValueExecutor);
  custom_executors.push_back(scatterValueInplaceExecutor);
  register_op(
      {/* op_name */ "scatter",
       /* overload */ "value",
       /* op_namespace */ "torch"},
      scatterValueExecutor.get());
  register_op(
      {/* op_name */ "scatter",
       /* overload */ "value",
       /* op_namespace */ "torch.Tensor"},
      scatterValueExecutor.get());
  register_op(
      {/* op_name */ "scatter_",
       /* overload */ "value",
       /* op_namespace */ "torch.Tensor"},
      scatterValueInplaceExecutor.get());
}

void SharedLayerReportGenerator::register_scatter_out_exception() {
  /* SRC */
  std::shared_ptr<IStackGenerator> scatterSrcStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          {3}));
  std::shared_ptr<SharedLayerExecutor<>> scatterSrcExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          scatterSrcStackGenerator.get(), &habana::validator_scatter_src_out);
  custom_stack_generators.push_back(scatterSrcStackGenerator);
  custom_executors.push_back(scatterSrcExecutor);
  register_op(
      {/* op_name */ "scatter",
       /* overload */ "src_out",
       /* op_namespace */ "torch"},
      scatterSrcExecutor.get());

  /* VALUE */
  std::shared_ptr<IStackGenerator> scatterValueStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          {3}));
  std::shared_ptr<SharedLayerExecutor<>> scatterValueExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          scatterValueStackGenerator.get(),
          &habana::validator_scatter_value_out);
  custom_stack_generators.push_back(scatterValueStackGenerator);
  custom_executors.push_back(scatterValueExecutor);
  register_op(
      {/* op_name */ "scatter",
       /* overload */ "value_out",
       /* op_namespace */ "torch"},
      scatterValueExecutor.get());
}

void SharedLayerReportGenerator::register_searchsorted_exception() {
  /* TENSOR */
  std::shared_ptr<IStackGenerator> searchSortedTensorStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor sorted_sequence, Tensor self, bool out_int32, bool right, str? side, Tensor? sorter",
          "searchsorted",
          "searchsorted.Tensor",
          std::vector<std::int64_t>{2});
  std::shared_ptr<SharedLayerExecutor<>> searchSortedTensorExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          searchSortedTensorStackGenerator.get(),
          &habana::validator_searchsorted_Tensor);
  custom_stack_generators.push_back(searchSortedTensorStackGenerator);
  custom_executors.push_back(searchSortedTensorExecutor);

  register_op(
      {/* op_name */ "searchsorted",
       /* overload */ "Tensor",
       /* op_namespace */ "torch"},
      searchSortedTensorExecutor.get());

  /* TENSOR_OUT */
  std::shared_ptr<IStackGenerator> searchSortedTensorOutStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor sorted_sequence, Tensor self, bool out_int32, bool right, str? side, Tensor? sorter, Tensor out",
          "searchsorted",
          "searchsorted.Tensor_out",
          std::vector<std::int64_t>{2});
  std::shared_ptr<SharedLayerExecutor<>> searchSortedTensorOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          searchSortedTensorOutStackGenerator.get(),
          &habana::validator_searchsorted_Tensor_out);
  custom_stack_generators.push_back(searchSortedTensorOutStackGenerator);
  custom_executors.push_back(searchSortedTensorOutExecutor);

  register_op(
      {/* op_name */ "searchsorted",
       /* overload */ "Tensor_out",
       /* op_namespace */ "torch"},
      searchSortedTensorOutExecutor.get());

  /* SCALAR */
  std::shared_ptr<IStackGenerator> searchSortedScalarStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor sorted_sequence, Scalar self, bool out_int32, bool right, str? side, Tensor? sorter",
          "searchsorted",
          "searchsorted.Scalar");
  std::shared_ptr<SharedLayerExecutor<>> searchSortedScalarExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          searchSortedScalarStackGenerator.get(),
          &habana::validator_searchsorted_Scalar);
  custom_stack_generators.push_back(searchSortedScalarStackGenerator);
  custom_executors.push_back(searchSortedScalarExecutor);

  register_op(
      {/* op_name */ "searchsorted",
       /* overload */ "Scalar",
       /* op_namespace */ "torch"},
      searchSortedScalarExecutor.get());

  /* SCALAR_OUT */
  std::shared_ptr<IStackGenerator> searchSortedScalarOutStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor sorted_sequence, Scalar self, bool out_int32, bool right, str? side, Tensor? sorter, Tensor out",
          "searchsorted",
          "searchsorted.Scalar_out");
  std::shared_ptr<SharedLayerExecutor<>> searchSortedScalarOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          searchSortedScalarOutStackGenerator.get(),
          &habana::validator_searchsorted_Scalar_out);
  custom_stack_generators.push_back(searchSortedScalarOutStackGenerator);
  custom_executors.push_back(searchSortedScalarOutExecutor);

  register_op(
      {/* op_name */ "searchsorted",
       /* overload */ "Scalar_out",
       /* op_namespace */ "torch"},
      searchSortedScalarOutExecutor.get());
}

void SharedLayerReportGenerator::register_max_unpool2d_exception() {
  std::shared_ptr<IStackGenerator> maxUnpool2dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, Tensor indices, SymInt[2] output_size",
          "max_unpool2d",
          "max_unpool2d",
          std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> maxUnpool2dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          maxUnpool2dStackGenerator.get(), &habana::validator_max_unpool2d);
  custom_stack_generators.push_back(maxUnpool2dStackGenerator);
  custom_executors.push_back(maxUnpool2dExecutor);
  register_op(
      {/* op_name */ "MaxUnpool2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      maxUnpool2dExecutor.get());
  register_op(
      {/* op_name */ "max_unpool2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      maxUnpool2dExecutor.get());
}

void SharedLayerReportGenerator::register_max_unpool2d_out_exception() {
  std::shared_ptr<IStackGenerator> maxUnpool2dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, Tensor indices, SymInt[2] output_size, Tensor out",
          "max_unpool2d",
          "max_unpool2d.out",
          std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> maxUnpool2dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          maxUnpool2dStackGenerator.get(), &habana::validator_max_unpool2d_out);
  custom_stack_generators.push_back(maxUnpool2dStackGenerator);
  custom_executors.push_back(maxUnpool2dExecutor);
  register_op(
      {/* op_name */ "max_unpool2d",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      maxUnpool2dExecutor.get());
}

void SharedLayerReportGenerator::register_max_unpool3d_exception() {
  std::shared_ptr<IStackGenerator> maxUnpool3dStackGenerator = std::make_shared<
      SchemaStackGenerator>(
      "Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding",
      "max_unpool3d",
      "max_unpool3d",
      std::vector<std::int64_t>{5});
  std::shared_ptr<SharedLayerExecutor<>> maxUnpool3dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          maxUnpool3dStackGenerator.get(), &habana::validator_max_unpool3d);
  custom_stack_generators.push_back(maxUnpool3dStackGenerator);
  custom_executors.push_back(maxUnpool3dExecutor);
  register_op(
      {/* op_name */ "MaxUnpool3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      maxUnpool3dExecutor.get());
  register_op(
      {/* op_name */ "max_unpool3d",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      maxUnpool3dExecutor.get());
}

void SharedLayerReportGenerator::register_max_unpool3d_out_exception() {
  std::shared_ptr<IStackGenerator> maxUnpool3dStackGenerator = std::make_shared<
      SchemaStackGenerator>(
      "Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding, Tensor out",
      "max_unpool3d",
      "max_unpool3d.out",
      std::vector<std::int64_t>{5});
  std::shared_ptr<SharedLayerExecutor<>> maxUnpool3dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          maxUnpool3dStackGenerator.get(), &habana::validator_max_unpool3d_out);
  custom_stack_generators.push_back(maxUnpool3dStackGenerator);
  custom_executors.push_back(maxUnpool3dExecutor);
  register_op(
      {/* op_name */ "max_unpool3d",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      maxUnpool3dExecutor.get());
}

void SharedLayerReportGenerator::register_upsample_exception() {
  /* BILINEAR */
  std::shared_ptr<IStackGenerator> upsampleBilinear2dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h, float? scales_w",
          "upsample_bilinear2d",
          "",
          std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> upsampleBilinear2dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          upsampleBilinear2dStackGenerator.get(),
          &habana::validator_upsample_bilinear2d);
  std::shared_ptr<SharedLayerExecutor<>> upsampleBilinear2dAaExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          upsampleBilinear2dStackGenerator.get(),
          &habana::validator__upsample_bilinear2d_aa);
  custom_stack_generators.push_back(upsampleBilinear2dStackGenerator);
  custom_executors.push_back(upsampleBilinear2dExecutor);
  custom_executors.push_back(upsampleBilinear2dAaExecutor);

  register_op(
      {/* op_name */ "Upsample",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      upsampleBilinear2dExecutor.get());
  register_op(
      {/* op_name */ "upsample",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      upsampleBilinear2dExecutor.get());
  register_op(
      {/* op_name */ "upsample_bilinear",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      upsampleBilinear2dAaExecutor.get());

  /* BICUBIC */
  std::shared_ptr<IStackGenerator> upsampleBicubic2dStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h, float? scales_w",
          "upsample_bicubic2d",
          "",
          std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> upsampleBicubic2dExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          upsampleBicubic2dStackGenerator.get(),
          &habana::validator_upsample_bilinear2d);
  custom_stack_generators.push_back(upsampleBicubic2dStackGenerator);
  custom_executors.push_back(upsampleBicubic2dExecutor);

  register_op(
      {/* op_name */ "Upsample",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      upsampleBicubic2dExecutor.get());
  register_op(
      {/* op_name */ "upsample",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      upsampleBicubic2dExecutor.get());
}

void SharedLayerReportGenerator::register_upsample_out_exception() {
  /* BILINEAR */
  std::shared_ptr<IStackGenerator> upsampleBilinear2dOutStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h, float? scales_w, Tensor out",
          "upsample_bilinear2d",
          "upsample_bilinear2d.out",
          std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> upsampleBilinear2dOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          upsampleBilinear2dOutStackGenerator.get(),
          &habana::validator_upsample_bilinear2d_out);
  std::shared_ptr<SharedLayerExecutor<>> upsampleBilinear2dAaOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          upsampleBilinear2dOutStackGenerator.get(),
          &habana::validator__upsample_bilinear2d_aa_out);
  custom_stack_generators.push_back(upsampleBilinear2dOutStackGenerator);
  custom_executors.push_back(upsampleBilinear2dOutExecutor);
  custom_executors.push_back(upsampleBilinear2dAaOutExecutor);

  register_op(
      {/* op_name */ "upsample",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      upsampleBilinear2dOutExecutor.get());
  register_op(
      {/* op_name */ "upsample_bilinear",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      upsampleBilinear2dAaOutExecutor.get());

  /* BICUBIC */
  std::shared_ptr<IStackGenerator> upsampleBicubic2dOutStackGenerator =
      std::make_shared<SchemaStackGenerator>(
          "Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h, float? scales_w, Tensor out",
          "upsample_bicubic2d",
          "upsample_bicubic2d.out",
          std::vector<std::int64_t>{4});
  std::shared_ptr<SharedLayerExecutor<>> upsampleBicubic2dOutExecutor =
      std::make_shared<GenericSharedLayerExecutor<>>(
          upsampleBicubic2dOutStackGenerator.get(),
          &habana::validator_upsample_bilinear2d_out);
  custom_stack_generators.push_back(upsampleBicubic2dOutStackGenerator);
  custom_executors.push_back(upsampleBicubic2dOutExecutor);

  register_op(
      {/* op_name */ "upsample",
       /* overload */ "out",
       /* op_namespace */ "torch.nn.functional"},
      upsampleBicubic2dOutExecutor.get());
}

void SharedLayerReportGenerator::register_where_exception() {
  std::shared_ptr<IStackGenerator> whereStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "where.self"));
  std::shared_ptr<SharedLayerExecutor<>> whereExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          whereStackGenerator.get(), &habana::validator_where_self);
  custom_stack_generators.push_back(whereStackGenerator);
  custom_executors.push_back(whereExecutor);
  register_op(
      {/* op_name */ "where",
       /* overload */ "self",
       /* op_namespace */ "torch"},
      whereExecutor.get());
  register_op(
      {/* op_name */ "where",
       /* overload */ "self",
       /* op_namespace */ "torch.Tensor"},
      whereExecutor.get());
}

void SharedLayerReportGenerator::register_where_out_exception() {
  std::shared_ptr<IStackGenerator> whereStackGenerator =
      std::make_shared<StackGenerator>(StackGenerator(
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
          "where.self_out"));
  std::shared_ptr<SharedLayerExecutor<>> whereExecutor =
      std::make_shared<CustomSharedLayerExecutor<>>(
          whereStackGenerator.get(), &habana::validator_where_self_out);
  custom_stack_generators.push_back(whereStackGenerator);
  custom_executors.push_back(whereExecutor);
  register_op(
      {/* op_name */ "where",
       /* overload */ "self_out",
       /* op_namespace */ "torch"},
      whereExecutor.get());
}

void SharedLayerReportGenerator::register_static_exceptions() {
  std::shared_ptr<SharedLayerExecutor<>> fpExceptFp8Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Half});
  custom_executors.push_back(fpExceptFp8Executor);
  std::shared_ptr<SharedLayerExecutor<>> fpExceptFp16Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Float8_e4m3fn,
              c10::ScalarType::Float8_e5m2});
  custom_executors.push_back(fpExceptFp16Executor);
  std::shared_ptr<SharedLayerExecutor<>> fp32Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{c10::ScalarType::Float});
  custom_executors.push_back(fp32Executor);
  std::shared_ptr<SharedLayerExecutor<>> fp32Bf16Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float, c10::ScalarType::BFloat16});
  custom_executors.push_back(fp32Bf16Executor);
  std::shared_ptr<SharedLayerExecutor<>> fp32Bf16I32Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Int});
  custom_executors.push_back(fp32Bf16I32Executor);
  std::shared_ptr<SharedLayerExecutor<>> fp32I32Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float, c10::ScalarType::Int});
  custom_executors.push_back(fp32I32Executor);
  std::shared_ptr<SharedLayerExecutor<>> fp32Bf16I32BoolExecutor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Int,
              c10::ScalarType::Bool});
  custom_executors.push_back(fp32Bf16I32BoolExecutor);
  std::shared_ptr<SharedLayerExecutor<>> fp32Bf16I32I8BoolExecutor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Int,
              c10::ScalarType::Char,
              c10::ScalarType::Bool});
  custom_executors.push_back(fp32Bf16I32I8BoolExecutor);
  std::shared_ptr<SharedLayerExecutor<>> fp32Bf16I32I8Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Int,
              c10::ScalarType::Char});
  custom_executors.push_back(fp32Bf16I32I8Executor);
  std::shared_ptr<SharedLayerExecutor<>> i32I16AndFpExceptFp16Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Float8_e4m3fn,
              c10::ScalarType::Float8_e5m2,
              c10::ScalarType::Int,
              c10::ScalarType::Short});
  custom_executors.push_back(i32I16AndFpExceptFp16Executor);
  std::shared_ptr<SharedLayerExecutor<>> i32AndFpExceptFp8Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Half,
              c10::ScalarType::Int});
  custom_executors.push_back(i32AndFpExceptFp8Executor);
  std::shared_ptr<SharedLayerExecutor<>> i32I16AndF32BF16Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Int,
              c10::ScalarType::Short});
  custom_executors.push_back(i32I16AndF32BF16Executor);
  std::shared_ptr<SharedLayerExecutor<>> allExecutor =
      std::make_shared<StaticSharedLayerExecutor<>>(
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
              c10::ScalarType::Bool});
  custom_executors.push_back(allExecutor);
  std::shared_ptr<SharedLayerExecutor<>> allExceptLongExecutor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Half,
              c10::ScalarType::Float8_e4m3fn,
              c10::ScalarType::Float8_e5m2,
              c10::ScalarType::Int,
              c10::ScalarType::Short,
              c10::ScalarType::Char,
              c10::ScalarType::Bool});
  custom_executors.push_back(allExceptLongExecutor);
  std::shared_ptr<SharedLayerExecutor<>> allExceptI8AndBoolExecutor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Half,
              c10::ScalarType::Float8_e4m3fn,
              c10::ScalarType::Float8_e5m2,
              c10::ScalarType::Long,
              c10::ScalarType::Int,
              c10::ScalarType::Short});
  custom_executors.push_back(allExceptI8AndBoolExecutor);
  std::shared_ptr<SharedLayerExecutor<>> allExceptFp8I16Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Half,
              c10::ScalarType::Long,
              c10::ScalarType::Int,
              c10::ScalarType::Char,
              c10::ScalarType::Bool});
  custom_executors.push_back(allExceptFp8I16Executor);
  std::shared_ptr<SharedLayerExecutor<>> i64I32Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Long, c10::ScalarType::Int});
  custom_executors.push_back(i64I32Executor);
  std::shared_ptr<SharedLayerExecutor<>> i32I8BoolExecutor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Int,
              c10::ScalarType::Char,
              c10::ScalarType::Bool});
  custom_executors.push_back(i32I8BoolExecutor);
  std::shared_ptr<SharedLayerExecutor<>> allIntegersExecutor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Long,
              c10::ScalarType::Int,
              c10::ScalarType::Short,
              c10::ScalarType::Char});
  custom_executors.push_back(allIntegersExecutor);
  std::shared_ptr<SharedLayerExecutor<>> allExceptFp16I64Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Float8_e4m3fn,
              c10::ScalarType::Float8_e5m2,
              c10::ScalarType::Int,
              c10::ScalarType::Short,
              c10::ScalarType::Char,
              c10::ScalarType::Bool});
  custom_executors.push_back(allExceptFp16I64Executor);
  std::shared_ptr<SharedLayerExecutor<>> allExceptFp16I64I16Executor =
      std::make_shared<StaticSharedLayerExecutor<>>(
          std::vector<c10::ScalarType>{
              c10::ScalarType::Float,
              c10::ScalarType::BFloat16,
              c10::ScalarType::Float8_e4m3fn,
              c10::ScalarType::Float8_e5m2,
              c10::ScalarType::Int,
              c10::ScalarType::Char,
              c10::ScalarType::Bool});
  custom_executors.push_back(allExceptFp16I64I16Executor);
  /* __AND__ */
  register_op(
      {/* op_name */ "__and__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor.get());

  /* BATCHED_NMS */
  register_op(
      {/* op_name */ "batched_nms",
       /* overload */ "",
       /* op_namespace */ "torchvision.ops"},
      fp32Bf16Executor.get());

  /* BATCH_NORM */
  register_op(
      {/* op_name */ "batch_norm",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "BatchNorm1d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "BatchNorm2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "batch_norm",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor.get());

  /* BINCOUNT */
  register_op(
      {/* op_name */ "bincount",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allIntegersExecutor.get());
  register_op(
      {/* op_name */ "bincount",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allIntegersExecutor.get());

  /* BROADCAST_TENSORS */
  register_op(
      {/* op_name */ "broadcast_tensors",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor.get());

  /* CHOLESKY */
  register_op(
      {/* op_name */ "cholesky",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Executor.get());
  register_op(
      {/* op_name */ "cholesky",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Executor.get());

  /* CHUNK */
  register_op(
      {/* op_name */ "chunk",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "chunk",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* CLIP */
  register_op(
      {/* op_name */ "clip",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "clip",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "clip_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "clip_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* CONJ */
  register_op(
      {/* op_name */ "conj",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "conj",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* CONSTANT_PAD_1D */
  register_op(
      {/* op_name */ "ConstantPad1d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor.get());

  /* COPY */
  register_op(
      {/* op_name */ "copy_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptFp16I64I16Executor.get());

  /* CROSS_ENTROPY_LOSS */
  register_op(
      {/* op_name */ "CrossEntropyLoss",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fp32Bf16Executor.get());

  /* DEFORM_CONV2D */
  register_op(
      {/* op_name */ "deform_conv2d",
       /* overload */ "",
       /* op_namespace */ "torchvision.ops"},
      fp32Executor.get());

  /* DIAG */
  register_op(
      {/* op_name */ "diag",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "diag",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fpExceptFp8Executor.get());

  /* DROPOUT*/
  register_op(
      {/* op_name */ "dropout", /* overload */ "", /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "Dropout",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "dropout",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor.get());

  /* EMBEDDING */
  register_op(
      {/* op_name */ "embedding_bag",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "EmbeddingBag",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "embedding_bag",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor.get());

  /* EMPTY */
  register_op(
      {/* op_name */ "empty",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor.get());
  register_op(
      {/* op_name */ "empty_like",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor.get());
  register_op(
      {/* op_name */ "empty_strided",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor.get());

  /* EXPAND_AS */
  register_op(
      {/* op_name */ "expand_as",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptFp8I16Executor.get());

  /* EXPIT */
  register_op(
      {/* op_name */ "expit",
       /* overload */ "",
       /* op_namespace */ "torch.special"},
      fp32Bf16Executor.get());

  /* FLATTEN */
  register_op(
      {/* op_name */ "flatten",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "flatten",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* FULL */
  register_op(
      {/* op_name */ "full",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor.get());
  register_op(
      {/* op_name */ "full_like",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor.get());

  /* __IAND__ */
  register_op(
      {/* op_name */ "__iand__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor.get());

  /* INDEX_ADD */
  register_op(
      {/* op_name */ "index_add_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32AndFpExceptFp8Executor.get());

  /* INDEX_PUT */
  register_op(
      {/* op_name */ "index_put",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "index_put_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "index_put",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "index_put_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* INSTANCE_NORM */
  register_op(
      {/* op_name */ "instance_norm",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "InstanceNorm2d",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "instance_norm",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fp32Bf16Executor.get());

  /* __IOR__ */
  register_op(
      {/* op_name */ "__ior__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor.get());

  /* IS_COMPLEX */
  register_op(
      {/* op_name */ "is_complex",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "is_complex",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* IS_FLOATING_POINT */
  register_op(
      {/* op_name */ "is_floating_point",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "is_floating_point",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* IS_NONZERO */
  register_op(
      {/* op_name */ "is_nonzero",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "is_nonzero",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* ITEM */
  register_op(
      {/* op_name */ "item",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Executor.get());

  /* __IXOR__ */
  register_op(
      {/* op_name */ "__ixor__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor.get());

  /* LAYER_NORM */
  register_op(
      {/* op_name */ "layer_norm",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "LayerNorm",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "layer_norm",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor.get());

  /* LOGSUMEXP */
  register_op(
      {/* op_name */ "logsumexp",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "logsumexp",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "logsumexp",
       /* overload */ "",
       /* op_namespace */ "torch.special"},
      fp32Bf16Executor.get());

  /* L1_LOSS */
  register_op(
      {/* op_name */ "l1_loss",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fp32Bf16Executor.get());

  /* MASKED_SELECT */
  register_op(
      {/* op_name */ "masked_select",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "masked_select",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* MATMUL */
  register_op(
      {/* op_name */ "matmul",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "matmul",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16Executor.get());

  /* MESHGRID */
  register_op(
      {/* op_name */ "meshgrid",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor.get());

  /* NARROW */
  register_op(
      {/* op_name */ "narrow",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Executor.get());
  register_op(
      {/* op_name */ "narrow",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Executor.get());

  /* _NATIVE_BATCH_NORM_LEGIT */
  register_op(
      {/* op_name */ "_native_batch_norm_legit",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());

  /* _NATIVE_BATCH_NORM_LEGIT_NO_TRAINING */
  register_op(
      {/* op_name */ "_native_batch_norm_legit_no_training",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());

  /* NATIVE_LAYER_NORM */
  register_op(
      {/* op_name */ "native_layer_norm",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());

  /* NEW_EMPTY */
  register_op(
      {/* op_name */ "new_empty",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8Executor.get());

  /* NEW_EMPTY_STRIDED */
  register_op(
      {/* op_name */ "new_empty_strided",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8Executor.get());

  /* NEW_FULL */
  register_op(
      {/* op_name */ "new_full",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8Executor.get());

  /* NEW_ONES */
  register_op(
      {/* op_name */ "new_ones",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8Executor.get());

  /* NMS */
  register_op(
      {/* op_name */ "nms",
       /* overload */ "",
       /* op_namespace */ "torchvision.ops"},
      fp32Bf16Executor.get());

  /* NONZERO */
  register_op(
      {/* op_name */ "nonzero",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32BoolExecutor.get());
  register_op(
      {/* op_name */ "nonzero",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32BoolExecutor.get());

  /* ONES */
  register_op(
      {/* op_name */ "ones",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptFp16I64I16Executor.get());

  /* ONES_LIKE */
  register_op(
      {/* op_name */ "ones_like",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptFp16I64Executor.get());

  /* __OR__ */
  register_op(
      {/* op_name */ "__or__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor.get());

  /* PAD */
  register_op(
      {/* op_name */ "pad",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp16Executor.get());

  /* PRELU */
  register_op(
      {/* op_name */ "prelu",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "PReLU",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "prelu",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fp32Bf16Executor.get());

  /* REPEAT_INTERLEAVE */
  register_op(
      {/* op_name */ "repeat_interleave",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptFp8I16Executor.get());

  /* RESHAPE */
  register_op(
      {/* op_name */ "reshape",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExecutor.get());
  register_op(
      {/* op_name */ "reshape",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExecutor.get());

  /* RESOLVE_CONJ */
  register_op(
      {/* op_name */ "resolve_conj",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "resolve_conj",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16Executor.get());

  /* RESOLVE_NEG */
  register_op(
      {/* op_name */ "resolve_neg",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16Executor.get());
  register_op(
      {/* op_name */ "resolve_neg",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16Executor.get());

  /* RESULT_TYPE */
  register_op(
      {/* op_name */ "result_type",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());

  /* RESULT_TYPE */
  register_op(
      {/* op_name */ "pin_memory",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8BoolExecutor.get());

  /* ROI_ALIGN */
  register_op(
      {/* op_name */ "roi_align",
       /* overload */ "",
       /* op_namespace */ "torchvision.ops"},
      fp32Executor.get());

  /* SPLIT_WITH_SIZES */
  register_op(
      {/* op_name */ "split_with_sizes",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i32I16AndFpExceptFp16Executor.get());
  register_op(
      {/* op_name */ "split_with_sizes",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I16AndFpExceptFp16Executor.get());

  /* SQUARE */
  register_op(
      {/* op_name */ "square",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32I8BoolExecutor.get());
  register_op(
      {/* op_name */ "square",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8BoolExecutor.get());
  register_op(
      {/* op_name */ "square_",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32I8BoolExecutor.get());
  register_op(
      {/* op_name */ "square_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32I8BoolExecutor.get());

  /* SOFTMAX */
  register_op(
      {/* op_name */ "softmax",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "Softmax",
       /* overload */ "",
       /* op_namespace */ "torch.nn"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "softmax",
       /* overload */ "",
       /* op_namespace */ "torch.nn.functional"},
      fpExceptFp8Executor.get());
  register_op(
      {/* op_name */ "softmax",
       /* overload */ "",
       /* op_namespace */ "torch.special"},
      fpExceptFp8Executor.get());

  /* STACK */
  register_op(
      {/* op_name */ "stack",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i32I16AndF32BF16Executor.get());

  /* T */
  register_op(
      {/* op_name */ "T",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExecutor.get());

  /* to */
  register_op(
      {/* op_name */ "to",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExecutor.get());

  /* TRIL_INDICES */
  register_op(
      {/* op_name */ "tril_indices",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i64I32Executor.get());

  /* TRIU_INDICES */
  register_op(
      {/* op_name */ "triu_indices",
       /* overload */ "",
       /* op_namespace */ "torch"},
      i64I32Executor.get());

  /* UNBIND */
  register_op(
      {/* op_name */ "unbind",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32Bf16I32Executor.get());
  register_op(
      {/* op_name */ "unbind",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32Bf16I32Executor.get());

  /* UNIQUE */
  register_op(
      {/* op_name */ "unique",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32I32Executor.get());
  register_op(
      {/* op_name */ "unique",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      fp32I32Executor.get());

  /* _UNIQUE */
  register_op(
      {/* op_name */ "_unique",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32I32Executor.get());

  /* _UNIQUE2 */
  register_op(
      {/* op_name */ "_unique2",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fp32I32Executor.get());

  /* UNSQUEEZE */
  register_op(
      {/* op_name */ "unsqueeze",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptI8AndBoolExecutor.get());
  register_op(
      {/* op_name */ "unsqueeze",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptI8AndBoolExecutor.get());
  register_op(
      {/* op_name */ "unsqueeze_",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      allExceptI8AndBoolExecutor.get());

  /* WEIGHT_NORM */
  register_op(
      {/* op_name */ "weight_norm",
       /* overload */ "",
       /* op_namespace */ "torch.nn.utils"},
      fp32Bf16Executor.get());

  /* _WEIGHT_NORM_INTERFACE */
  register_op(
      {/* op_name */ "_weight_norm_interface",
       /* overload */ "",
       /* op_namespace */ "torch"},
      fpExceptFp8Executor.get());

  /* __XOR__ */
  register_op(
      {/* op_name */ "__xor__",
       /* overload */ "",
       /* op_namespace */ "torch.Tensor"},
      i32I8BoolExecutor.get());

  /* ZEROS */
  register_op(
      {/* op_name */ "zeros",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptLongExecutor.get());

  /* ZEROS_LIKE */
  register_op(
      {/* op_name */ "zeros_like",
       /* overload */ "",
       /* op_namespace */ "torch"},
      allExceptLongExecutor.get());
}

} // namespace slrg
