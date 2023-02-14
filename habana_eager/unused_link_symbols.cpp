
/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "habana_eager/helpers.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
#include "habana_lazy/hpu_lazy_tensors.h"

using namespace at;

// *************************************************
// This file contains list of symbols needed to link new frontend plugin, but
// not relevant for eager execution. They will be removed once backend
// dependencies are cleared out as a part of SW-123330

// It also contains list of manual/override_fn ops included in
// wrap_kernel_register, that are mandatory for PT2.0, but not implemented for
// new frontend. They all will be moved to backend as a part of SW-118176
// *************************************************

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::batch_norm_stats(
    const at::Tensor&,
    double) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor hpu_wrap::batch_norm_elemt(
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const at::Tensor&,
    const at::Tensor&,
    double) {
  EAGER_NOT_SUPPORTED;
}

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::
    batch_norm_gather_stats_with_counts(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const c10::optional<at::Tensor>&,
        const c10::optional<at::Tensor>&,
        double,
        double,
        const at::Tensor&) {
  EAGER_NOT_SUPPORTED;
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hpu_wrap::
    batch_norm_backward_reduce(
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const at::Tensor&,
        const c10::optional<at::Tensor>&,
        bool,
        bool,
        bool) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor hpu_wrap::batch_norm_backward_elemt(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  EAGER_NOT_SUPPORTED;
}

Tensor hpu_wrap::_pin_memory(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor hpu_wrap::repeat_interleave(
    const at::Tensor&,
    c10::optional<int64_t>) {
  EAGER_NOT_SUPPORTED;
}

Tensor hpu_wrap::_efficientzerotensor(
    IntArrayRef,
    c10::optional<ScalarType>,
    c10::optional<Layout>,
    c10::optional<Device>,
    c10::optional<bool>) {
  EAGER_NOT_SUPPORTED;
}

Tensor& hpu_wrap::index_add_out(
    const Tensor&,
    int64_t,
    const Tensor&,
    const Tensor&,
    const Scalar&,
    Tensor&) {
  EAGER_NOT_SUPPORTED;
}

Tensor& hpu_wrap::index_fill_(Tensor&, int64_t, const Tensor&, const Scalar&) {
  EAGER_NOT_SUPPORTED;
}

Tensor& hpu_wrap::masked_select_out(const Tensor&, const Tensor&, Tensor&) {
  EAGER_NOT_SUPPORTED;
}

Tensor hpu_wrap::masked_select(const Tensor&, const Tensor&) {
  EAGER_NOT_SUPPORTED;
}

Tensor& hpu_wrap::nonzero_out(const Tensor&, Tensor&) {
  EAGER_NOT_SUPPORTED;
}

Tensor& hpu_wrap::max_pool2d_with_indices_backward_out(
    const Tensor&,
    const Tensor&,
    IntArrayRef,
    IntArrayRef,
    IntArrayRef,
    IntArrayRef,
    bool,
    const Tensor&,
    Tensor&) {
  EAGER_NOT_SUPPORTED;
}

// *************************************************
// BELOW is list of symbols needed to link new frontend plugin but not relevant
// for eager execution. They will be removed once backend dependencies
// are cleared out as a part of SW-123330
// *************************************************

namespace habana_lazy {

at::Tensor squeeze_hpu_lazy(const at::Tensor&, const int64_t) {
  EAGER_NOT_SUPPORTED;
}

std::vector<at::Tensor> split_with_sizes_hpu_lazy(
    const at::Tensor&,
    at::IntArrayRef,
    int64_t) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor nonzero_hpu_lazy(const at::Tensor&) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor append_to_batch_h2d_list(const at::Tensor&) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor get_tensor_for_scalar(double, const at::TensorOptions&) {
  EAGER_NOT_SUPPORTED;
}

void flush_op(
    size_t,
    std::shared_ptr<HbLazyFrontEndInfoToBackend>,
    std::vector<HbLazyTensor>) {
  EAGER_NOT_SUPPORTED;
}

void handle_collective(const at::IValue&) {
  EAGER_NOT_SUPPORTED;
}

Tensor empty_as_strided_lazy(
    const Tensor&,
    IntArrayRef,
    IntArrayRef,
    c10::optional<int64_t>) {
  EAGER_NOT_SUPPORTED;
}

ir::NodePtr create_as_strided_node(
    at::Tensor const&,
    c10::ArrayRef<long>,
    c10::ArrayRef<long>,
    c10::optional<long>,
    bool) {
  EAGER_NOT_SUPPORTED;
}

bool is_inplace(at::Symbol) {
  EAGER_NOT_SUPPORTED;
}

void strided_insert_hpu_lazy(const Tensor&, const Tensor&, bool) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& set_source_Storage_storage_offset(
    at::Tensor&,
    at::Storage,
    int64_t,
    at::IntArrayRef,
    at::IntArrayRef) {
  EAGER_NOT_SUPPORTED;
}

Tensor empty_strided_hpu_lazy(
    IntArrayRef,
    IntArrayRef,
    const TensorOptions&,
    bool,
    synTensorType,
    int64_t,
    c10::optional<std::reference_wrapper<const at::Tensor>>,
    bool) {
  EAGER_NOT_SUPPORTED;
}

void InitSizesAndStrides(
    at::Tensor&,
    c10::optional<synTensorType>,
    c10::optional<IntArrayRef>,
    c10::optional<IntArrayRef>,
    c10::optional<MemoryFormat>) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& broadcast_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] int64_t root_rank,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& allreduce_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] uint8_t reduce_op,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& reduce_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] int64_t dst_rank,
    [[maybe_unused]] uint8_t reduce_op,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& alltoall_hpu_lazy_out(
    [[maybe_unused]] const at::Tensor& input_tensor,
    [[maybe_unused]] int64_t comm_id,
    [[maybe_unused]] at::Tensor& output_tensor,
    [[maybe_unused]] std::vector<int64_t>& outputSplitSizes,
    [[maybe_unused]] std::vector<int64_t>& inputSplitSizes) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& allgather_hpu_lazy_out(
    [[maybe_unused]] const at::Tensor& inputTensor,
    [[maybe_unused]] int64_t comm_id,
    [[maybe_unused]] at::Tensor& output_tensor) {
  EAGER_NOT_SUPPORTED;
}
at::Tensor& reduce_scatter_hpu_lazy_out(
    [[maybe_unused]] const at::Tensor& input_tensor,
    [[maybe_unused]] uint8_t reduce_op,
    [[maybe_unused]] int64_t comm_id,
    [[maybe_unused]] at::Tensor& output_tensor) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& send_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] int64_t dst_rank,
    [[maybe_unused]] int64_t tag,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& recv_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] int64_t src_rank,
    [[maybe_unused]] int64_t tag,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

} // namespace habana_lazy
