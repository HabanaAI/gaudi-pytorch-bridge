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
#include "HPUStream.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace at {
namespace hpu {

struct SingleHPUGraph {
  SingleHPUGraph(
      std::shared_ptr<torch::jit::Graph> graph,
      habana_lazy::ir::ValueList input_vals,
      habana_lazy::ir::ValueList output_vals,
      std::vector<habana_lazy::HbLazyTensor> hblazy_tensors,
      size_t hash,
      size_t graphKey,
      std::string opStrs)
      : graph_{graph},
        input_vals_{input_vals},
        output_vals_{output_vals},
        hblazy_tensors_{hblazy_tensors},
        hash_{hash},
        graphKey_{graphKey},
        opStrs_{opStrs} {}

  ~SingleHPUGraph();
  void replay();
  std::shared_ptr<torch::jit::Graph> graph_;
  habana_lazy::ir::ValueList input_vals_;
  habana_lazy::ir::ValueList output_vals_;
  std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_;
  size_t hash_{0};
  size_t graphKey_{0};
  std::string opStrs_ = "";
};

struct HPUGraph {
  HPUGraph();
  ~HPUGraph();

  void capture_begin();
  void capture_end();
  void replay();
  void mark_step();

 protected:
  // Stream on which capture began
  c10::hpu::HPUStream capture_stream_;
  std::recursive_mutex mutex_;
  bool capturing_ = false;
  std::vector<std::shared_ptr<SingleHPUGraph>> captured_graphs;
};

} // namespace hpu
} // namespace at
