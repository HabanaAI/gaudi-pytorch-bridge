/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once
#include <unordered_set>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/ir/ir.h>

#include "habana_helpers/misc_utils.h"
#include "habana_helpers/tensor_utils.h"
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
  size_t launch_counter;
  bool dynamic_shape;
};

struct LaunchEagerInfo {
  std::shared_ptr<HbLazyFrontEndInfoToBackend> lazyFrontEndInfo;
  std::vector<at::Tensor> retained_tensor_list;
  std::shared_ptr<habana_lazy::OptimizedJITGraphAndMetaData>
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
