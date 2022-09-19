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
#include "habana_lazy/hlexec.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace at {
namespace hpu {

struct HPUGraph {
  HPUGraph();
  ~HPUGraph();

  void capture_begin();
  void capture_end();
  void replay();

 protected:
  // Stream on which capture began
  c10::hpu::HPUStream capture_stream_;
  std::mutex mutex_;
  bool capturing_ = false;
  std::shared_ptr<torch::jit::Graph> graph_;
  habana_lazy::ir::ValueList input_vals_;
  habana_lazy::ir::ValueList output_vals_;
  std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_;
  size_t hash_{0};
  size_t graphKey_{0};
  std::string opStrs_ = "";
};

} // namespace hpu
} // namespace at
