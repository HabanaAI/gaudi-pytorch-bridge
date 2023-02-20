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

#pragma once
#include <unordered_set>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/ir/ir.h>

#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/misc_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "ir_utils.h"

namespace habana_lazy {
struct LaunchTensorsInfo {
  std::vector<HbLazyTensor> tensors_ptr;
  std::vector<std::shared_ptr<Data>> input_list;
  std::vector<int> indices;
  // Tensorids list which is part of current exec thread
  std::vector<int64_t> executing_tids;
  habana_lazy::ir::PostOrderData po_data;
  exec::HlExec hlexec;
  torch::jit::Stack stack;
  bool async;
  bool has_queued;
  uint64_t launch_jobid;
  bool dynamic_shape;
};

struct LaunchEagerInfo {
  std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo;
  std::vector<at::Tensor> retained_tensor_list;
  std::shared_ptr<habana::OptimizedJITGraphAndMetaData>
      optimized_path_jit_ir_and_mdata;
  std::string lazyOpName;
  std::vector<std::vector<int64_t>> out_shapes;
  size_t optimizedLazyEagerKey;
  bool isOptimizedLazyEager;
};

struct LaunchStreamInfo {
  const c10::hpu::HPUStream stream;
  synEventHandle event_handle;
  synapse_helpers::hpuStream_t event_stream;
  bool event_flag;
};

} // namespace habana_lazy
