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
      std::unordered_map<int64_t, c10::optional<at::Generator>>
          seed_tensors_generator,
      size_t hash,
      size_t graphKey,
      std::string opStrs)
      : graph_{graph},
        input_vals_{input_vals},
        output_vals_{output_vals},
        hblazy_tensors_out_{hblazy_tensors},
        seed_tensors_generator_{seed_tensors_generator},
        hash_{hash},
        graphKey_{graphKey},
        opStrs_{opStrs} {}

  ~SingleHPUGraph();
  void replay(bool async = false);
  void replayV2(
      std::vector<at::Tensor>& static_inputs,
      std::vector<at::Tensor>& inputs,
      bool async = false);
  void replayV3(std::vector<at::Tensor>& outputs, bool async = false);
  void mark_user_outputs(std::vector<at::Tensor>& outputs);
  void replayGraph(habana_lazy::ir::ValueList& input_vals, bool async = false);

  std::shared_ptr<torch::jit::Graph> graph_;
  habana_lazy::ir::ValueList input_vals_;
  habana_lazy::ir::ValueList output_vals_;
  std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_in_;
  std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_out_;
  std::vector<std::pair<size_t, size_t>> user_out_indices_tlist_;
  std::set<size_t> hpugraph_dependant_out_t_list_;

  std::unordered_map<int64_t, c10::optional<at::Generator>>
      seed_tensors_generator_;
  size_t hash_{0};
  size_t graphKey_{0};
  std::string opStrs_ = "";
};

struct HPUGraph {
  HPUGraph();
  ~HPUGraph();

  void capture_begin();
  void capture_end();
  void replay(bool async = false);
  void mark_step();
  void replayV2(
      std::vector<at::Tensor>& static_inputs,
      std::vector<at::Tensor>& inputs,
      bool async = false);
  void replayV3(std::vector<at::Tensor>& inputs, bool async = false);
  void mark_user_outputs(std::vector<at::Tensor>& outputs);
  void find_output_tensors();

 protected:
  // Stream on which capture began
  c10::hpu::HPUStream capture_stream_;
  std::recursive_mutex mutex_;
  bool dynamic_env_ = false;
  bool capturing_ = false;
  std::vector<std::shared_ptr<SingleHPUGraph>> captured_graphs;
};

} // namespace hpu
} // namespace at
