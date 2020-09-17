/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"

using namespace torch;
using namespace at;

static auto
    registry =
        torch::
            RegisterOperators()
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(copy_hpu_wrap_),
                            &copy_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(as_strided_hpu_wrap),
                            &as_strided_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::set_.source_Storage_storage_offset( Tensor(a !) self, Storage source, int storage_offset, int[] size, int[] stride = []) ->Tensor(a !)")
                        .impl_unboxedOnlyKernel<
                            decltype(set_hpu_wrap_),
                            &set_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(view_hpu_wrap),
                            &view_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(addcmul_hpu_wrap),
                            &addcmul_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::addcmul_(Tensor(a !) self, Tensor tensor1, Tensor tensor2, *, Scalar value = 1) -> Tensor(a !)")
                        .impl_unboxedOnlyKernel<
                            decltype(addcmul_hpu_wrap_),
                            &addcmul_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(addcdiv_hpu_wrap),
                            &addcdiv_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(addcdiv_hpu_wrap_),
                            &addcdiv_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(add_tensor_hpu_wrap_),
                            &add_tensor_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(add_tensor_hpu_wrap),
                            &add_tensor_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha = 1) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(add_scalar_hpu_wrap),
                            &add_scalar_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(add_scalar_hpu_wrap_),
                                    &add_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(sub_tensor_hpu_wrap),
                                    &sub_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(sub_tensor_hpu_wrap_),
                                    &sub_tensor_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(sub_scalar_hpu_wrap),
                                    &sub_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(sub_scalar_hpu_wrap_),
                                    &sub_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(mul_tensor_hpu_wrap_),
                                    &mul_tensor_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(mul_tensor_hpu_wrap),
                                    &mul_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(mul_scalar_hpu_wrap),
                                    &mul_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(mul_scalar_hpu_wrap_),
                                    &mul_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_tensor_hpu_wrap),
                                    &div_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_tensor_hpu_wrap_out),
                                    &div_tensor_hpu_wrap_out>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div_.Tensor(Tensor(a!) self, Tensor other) -> (Tensor(a!))")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_tensor_hpu_wrap_),
                                    &div_tensor_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_scalar_hpu_wrap),
                                    &div_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::div_.Scalar(Tensor(a!) self, Scalar other) -> (Tensor(a!))")
                                .impl_unboxedOnlyKernel<
                                    decltype(div_scalar_hpu_wrap_),
                                    &div_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_tensor_tensor_hpu_wrap),
                                    &pow_tensor_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_tensor_tensor_hpu_wrap_),
                                    &pow_tensor_tensor_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_tensor_scalar_hpu_wrap),
                                    &pow_tensor_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow_.Scalar(Tensor(a !) self, Scalar exponent) -> Tensor(a !)")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_tensor_scalar_hpu_wrap_),
                                    &pow_tensor_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::pow.Scalar(Scalar self, Tensor exponent)->Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(pow_scalar_tensor_hpu_wrap),
                                    &pow_scalar_tensor_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(rsub_scalar_hpu_wrap),
                                    &rsub_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(gt_hpu_wrap),
                                    &gt_hpu_wrap>(DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(eq_tensor_hpu_wrap),
                                    &eq_tensor_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(eq_tensor_out_hpu_wrap),
                                    &eq_tensor_out_hpu_wrap>(DispatchKey::
                                                                 HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(eq_tensor_scalar_hpu_wrap),
                                    &eq_tensor_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(lt_scalar_hpu_wrap),
                                    &lt_scalar_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(lt_tensor_hpu_wrap),
                                    &lt_tensor_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(convolution_hpu_wrap),
                                    &convolution_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")
                                .impl_unboxedOnlyKernel<
                                    decltype(convolution_backward_hpu_wrap),
                                    &convolution_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_bag_hpu_wrap),
                                    &embedding_bag_hpu_wrap>(DispatchKey::
                                                                 HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_bag_bwd_hpu_wrap),
                                    &embedding_bag_bwd_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(constant_pad_hpu_wrap),
                                    &constant_pad_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_hpu_wrap),
                                    &embedding_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_dense_backward_hpu_wrap),
                                    &embedding_dense_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::embedding_bag_sum_fwd(Tensor input, Tensor indices_fwd, Tensor offsets_fwd, Tensor valid_count_fwd, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, Tensor grad_weight) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(embedding_bag_sum_fwd_hpu_wrap),
                                    &embedding_bag_sum_fwd_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::embedding_bag_sum_bwd.out(Tensor input, Tensor indices_bwd, Tensor offsets_bwd, Tensor valid_count_bwd, *, Tensor(a!) out) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(
                                        embedding_bag_sum_bwd_out_hpu_wrap),
                                    &embedding_bag_sum_bwd_out_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(fill_hpu_wrap_),
                                    &fill_hpu_wrap_>(DispatchKey::
                                                         HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(masked_fill_hpu_wrap_),
                                    &masked_fill_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(masked_fill_scalar_hpu_wrap_),
                                    &masked_fill_scalar_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(index_select_hpu_wrap),
                                    &index_select_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(index_put_hpu_wrap_),
                                    &index_put_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(index_put_hpu_wrap),
                                    &index_put_hpu_wrap>(DispatchKey::
                                                             HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(index_add_hpu_wrap_),
                                    &index_add_hpu_wrap_>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(scatter_inplace_src_hpu_wrap),
                                    &scatter_inplace_src_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(scatter_src_hpu_wrap),
                                    &scatter_src_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(gather_src_hpu_wrap),
                                    &gather_src_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(scatter_add_src_hpu_wrap),
                                    &scatter_add_src_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(scatter_add_inplace_src_hpu_wrap),
                                    &scatter_add_inplace_src_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::slice.Tensor(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)")
                                .impl_unboxedOnlyKernel<
                                    decltype(slice_hpu_wrap),
                                    &slice_hpu_wrap>(DispatchKey::
                                                         HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)")
                                .impl_unboxedOnlyKernel<
                                    decltype(select_hpu_wrap),
                                    &select_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(arange_hpu_wrap),
                                    &arange_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mm(Tensor self, Tensor mat2) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(mm_hpu_wrap),
                                    &mm_hpu_wrap>(DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta = 1, Scalar alpha = 1) ->Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(addmm_hpu_wrap),
                                    &addmm_hpu_wrap>(DispatchKey::
                                                         HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(batch_gemm_out_hpu_wrap),
                                    &batch_gemm_out_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::bmm(Tensor self, Tensor mat2) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(batch_gemm_hpu_wrap),
                                    &batch_gemm_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::dot(Tensor self, Tensor tensor) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(dot_hpu_wrap),
                                    &dot_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mv(Tensor self, Tensor vec)->Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(mv_hpu_wrap),
                                    &mv_hpu_wrap>(DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) ->(Tensor output, Tensor total_weight)")
                                .impl_unboxedOnlyKernel<
                                    decltype(nll_loss_forward_hpu_wrap),
                                    &nll_loss_forward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(nll_loss_backward_hpu_wrap),
                                    &nll_loss_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(mse_loss_forward_hpu_wrap),
                                    &mse_loss_forward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(mse_loss_backward_hpu_wrap),
                                    &mse_loss_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(binary_cross_entropy_hpu_wrap),
                                    &binary_cross_entropy_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(
                                        binary_cross_entropy_backward_hpu_wrap),
                                    &binary_cross_entropy_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")
                                .impl_unboxedOnlyKernel<
                                    decltype(batch_norm_hpu_wrap),
                                    &batch_norm_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
                                .impl_unboxedOnlyKernel<
                                    decltype(batch_norm_bwd_hpu_wrap),
                                    &batch_norm_bwd_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)")
                                .impl_unboxedOnlyKernel<
                                    decltype(layer_norm_hpu_wrap),
                                    &layer_norm_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")
                                .impl_unboxedOnlyKernel<
                                    decltype(layer_norm_backward_hpu_wrap),
                                    &layer_norm_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(norm_scalar_hpu_wrap),
                                    &norm_scalar_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride = [], int[2] padding = 0, int[2] dilation = 1, bool ceil_mode = False) ->(Tensor, Tensor)")
                                .impl_unboxedOnlyKernel<
                                    decltype(max_pool2d_with_indices_hpu_wrap),
                                    &max_pool2d_with_indices_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(
                                        max_pool2d_with_indices_backward_hpu_wrap),
                                    &max_pool2d_with_indices_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")
                                .impl_unboxedOnlyKernel<
                                    decltype(
                                        max_pool2d_with_indices_backward_out_hpu_wrap),
                                    &max_pool2d_with_indices_backward_out_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(avg_pool2d_hpu_wrap),
                                    &avg_pool2d_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::
                        RegisterOperators::
                            options()
                                .schema(
                                    "aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor")
                                .impl_unboxedOnlyKernel<
                                    decltype(avg_pool2d_backward_hpu_wrap),
                                    &avg_pool2d_backward_hpu_wrap>(
                                    DispatchKey::HABANATensorId)
                                .aliasAnalysis(
                                    c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(uniform_hpu_wrap),
                                &uniform_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(normal_hpu_wrap),
                                &normal_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(bernoulli_hpu_wrap),
                                &bernoulli_hpu_wrap>(DispatchKey::
                                                         HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(bernoulli_scalar_hpu_wrap),
                                &bernoulli_scalar_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(sum_dim_IntList_hpu_wrap),
                                &sum_dim_IntList_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(sum_IntList_out_hpu_wrap),
                                &sum_IntList_out_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(mean_dim_hpu_wrap),
                                &mean_dim_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(mean_dim_out_hpu_wrap),
                                &mean_dim_out_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(sum_hpu_wrap),
                                &sum_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(mean_hpu_wrap),
                                &mean_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(any_dim_hpu_wrap),
                                &any_dim_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema("aten::any(Tensor self) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(any_hpu_wrap),
                                &any_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(any_dim_out_hpu_wrap),
                                &any_dim_out_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::_local_scalar_dense(Tensor self) -> Scalar")
                            .impl_unboxedOnlyKernel<
                                decltype(
                                    at::native::_local_scalar_dense_hpu_wrap),
                                &at::native::_local_scalar_dense_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(log_softmax_hpu_wrap),
                                &log_softmax_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(habana::log_softmax_backward_hpu_wrap),
                                &habana::log_softmax_backward_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(habana::softmax_hpu_wrap),
                                &habana::softmax_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(habana::softmax_backward_hpu_wrap),
                                &habana::softmax_backward_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(clone_hpu_wrap),
                                &clone_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(at::native::empty_hpu_wrap),
                                &at::native::empty_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(at::native::empty_strided_hpu_wrap),
                                &at::native::empty_strided_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::zero_(Tensor(a!) self) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(zero_hpu_wrap),
                                &zero_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)")
                            .impl_unboxedOnlyKernel<
                                decltype(permute_hpu_wrap),
                                &permute_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)")
                            .impl_unboxedOnlyKernel<
                                decltype(expand_hpu_wrap),
                                &expand_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::cat(Tensor[] tensors, int dim=0) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(cat_hpu_wrap),
                                &cat_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(cat_hpu_wrap_out),
                                &cat_hpu_wrap_out>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::_cat(Tensor[] tensors, int dim=0) -> Tensor")
                            .impl_unboxedOnlyKernel<
                                decltype(cat_hpu_wrap),
                                &cat_hpu_wrap>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")
                            .impl_unboxedOnlyKernel<
                                decltype(cat_hpu_wrap_out),
                                &cat_hpu_wrap_out>(DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]")
                            .impl_unboxedOnlyKernel<
                                decltype(split_with_sizes_hpu_wrap),
                                &split_with_sizes_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::
                        options()
                            .schema(
                                "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")
                            .impl_unboxedOnlyKernel<
                                decltype(transpose_hpu_wrap),
                                &transpose_hpu_wrap>(
                                DispatchKey::HABANATensorId)
                            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(transpose_hpu_wrap_),
                            &transpose_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::t(Tensor(a) self) -> Tensor(a)")
                        .impl_unboxedOnlyKernel<
                            decltype(t_hpu_wrap),
                            &t_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::t_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(t_hpu_wrap_),
                            &t_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(threshold_backward_hpu_wrap),
                            &threshold_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)")
                        .impl_unboxedOnlyKernel<
                            decltype(topk_hpu_wrap),
                            &topk_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)")
                        .impl_unboxedOnlyKernel<
                            decltype(topk_out_hpu_wrap),
                            &topk_out_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)")
                        .impl_unboxedOnlyKernel<
                            decltype(sort_hpu_wrap),
                            &sort_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::relu_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(relu_hpu_wrap_),
                            &relu_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::relu(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(relu_hpu_wrap),
                            &relu_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::sigmoid(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sigmoid_hpu_wrap),
                            &sigmoid_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sigmoid_backward_hpu_wrap),
                            &sigmoid_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::sqrt(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(sqrt_hpu_wrap),
                            &sqrt_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::tanh(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(tanh_hpu_wrap),
                            &tanh_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::tanh_backward(Tensor grad_output, Tensor output)->Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(tanh_backward_hpu_wrap),
                            &tanh_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::tanh_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(tanh_hpu_wrap_),
                            &tanh_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(tanh_out_hpu_wrap),
                            &tanh_out_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::gelu(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(gelu_hpu_wrap),
                            &gelu_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::gelu_backward(Tensor grad, Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(gelu_backward_hpu_wrap),
                            &gelu_backward_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::erf_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(erf_hpu_wrap_),
                            &erf_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::erf(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(erf_hpu_wrap),
                            &erf_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::exp_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(exp_hpu_wrap_),
                            &exp_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::exp(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(exp_hpu_wrap),
                            &exp_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(neg_out_hpu_wrap),
                            &neg_out_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(reciprocal_hpu_wrap_),
                            &reciprocal_hpu_wrap_>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::reciprocal(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(reciprocal_hpu_wrap),
                            &reciprocal_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")
                        .impl_unboxedOnlyKernel<
                            decltype(reciprocal_out_hpu_wrap),
                            &reciprocal_out_hpu_wrap>(
                            DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema(
                            "aten::clamp_min(Tensor self, Scalar min) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(clamp_min_hpu_wrap),
                            &clamp_min_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::abs(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(abs_hpu_wrap),
                            &abs_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
                .op(torch::RegisterOperators::options()
                        .schema("aten::neg(Tensor self) -> Tensor")
                        .impl_unboxedOnlyKernel<
                            decltype(neg_hpu_wrap),
                            &neg_hpu_wrap>(DispatchKey::HABANATensorId)
                        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
