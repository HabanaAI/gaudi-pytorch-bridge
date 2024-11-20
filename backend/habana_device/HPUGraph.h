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
#pragma once
#include "HPUStream.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace at {
namespace hpu {

struct SingleHPUGraph {
  SingleHPUGraph(
      std::shared_ptr<habana::RecipeArgumentSpec> cached_rarg_psh,
      std::shared_ptr<torch::jit::Graph> graph,
      habana_lazy::ir::ValueList input_vals,
      habana_lazy::ir::ValueList output_vals,
      std::vector<habana_lazy::HbLazyTensor> hblazy_tensors,
      std::unordered_map<size_t, size_t> user_input_indices,
      std::unordered_map<int64_t, c10::optional<at::Generator>>
          seed_tensors_generator,
      size_t hash,
      size_t graphKey,
      std::string opStrs)
      : cached_rarg_psh_{cached_rarg_psh},
        graph_{graph},
        input_vals_{input_vals},
        output_vals_{output_vals},
        hblazy_tensors_out_{hblazy_tensors},
        user_input_indices_{user_input_indices},
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
  void replayV3(std::vector<at::Tensor>& inputs, bool async = false);
  void replayGraph(habana_lazy::ir::ValueList& input_vals, bool async = false);

  std::shared_ptr<habana::RecipeArgumentSpec> cached_rarg_psh_{nullptr};
  std::shared_ptr<torch::jit::Graph> graph_;
  habana_lazy::ir::ValueList input_vals_;
  habana_lazy::ir::ValueList output_vals_;
  std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_in_;
  std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_out_;
  std::vector<habana_lazy::HbLazyTensor> prev_graph_interdep_out_t_list_;
  std::vector<std::pair<size_t, size_t>> user_out_indices_tlist_;

  /**
   * @brief
   * Handling Input and Output Tensors
   *
   * The SingleHPUGraph input can be of the following types -
   * 1. User provided input: These are inputs that will be provided anew by the
   * user on every replayV3. These tensors are enlisted with the HPUGraph with
   * mark_user_inputs.
   * 2. Graph specific inputs: Weight tensors, or scalars converted to tensors
   * such inputs. These are typically persistent across replays and it is fine
   * to keep additional references to them here.
   * 3. From previous SingleHPUGraph: An HPUGraph has a series of SingleHPUGraph
   * that are replayed in sequence. The input of a SingleHPUGraph can come from
   * the output of a previous SingleHPUGraph in this sequence.
   *
   * Out of these type 1 and 3 need not be retained by the SingleHPUGraph, as
   * they will be provided anew everytime. For type 1 and type 3 tensors, the
   * place of these inputs in the launched JIT graph is noted and they are
   * filled up on every replayV3 either from user inputs or with output coming
   * from a previous SingleHPUGraph in the sequence.
   *
   * The SingleHPUGraph output can be of the following types -
   * 1. User visible output: These are outputs that will be going back to the
   * user finally. These tensors can be enlisted with the HPU graph with an API
   * mark_user_outputs.
   * 2. Used by a SingleHPUGraph: The output of a SingleHPUGraph can go as the
   * type 3 input to an SingleHPUGraph.
   * 3. Output to python: These could be output that goes back to python HPU
   * grah capture context, but expires in python context before going to the
   * user. Hence, these are not marked as type 1 output.
   *
   * Out of these, only type 1 output need to be sent to the user beyond a
   * replay call frm python. Type 3 needs to be sent back to python code too,
   * but they can be deleted by the python context. The type 2 output tensors
   * are consumed within the SingleHPUGraph sequence entirely, hence these can
   * be cleaned after the use within the sequence of SingleHPUGraph.
   *
   * Once an output is no longer required after a JIT graph launch of the
   * SingleHPUGraph that is last user of the tensor, the lazy tensor sets the
   * data_ptr to an empty tensor to release the memory.
   *
   */
  // user_input_indices_ : JIT graph input idx -> user provided input idx
  std::unordered_map<size_t, size_t> user_input_indices_;
  std::unordered_map<size_t, size_t> user_input_view_indices_;
  std::unordered_map<int64_t, c10::optional<at::Generator>>
      seed_tensors_generator_;
  size_t hash_{0};
  size_t graphKey_{0};
  std::string opStrs_ = "";
};

struct HPUGraph {
  HPUGraph();
  ~HPUGraph();

  void capture_begin(bool dry_run = false);
  void capture_end();
  void replay(bool async = false);
  void mark_step();
  void clear_inputs();
  void replayV2(
      std::vector<at::Tensor>& static_inputs,
      std::vector<at::Tensor>& inputs,
      bool async = false);
  void replayV3(std::vector<at::Tensor>& inputs, bool async = false);
  void mark_user_outputs(std::vector<at::Tensor>& outputs);
  void mark_user_inputs(std::vector<at::Tensor>& static_inputs);
  void destroy();
  std::unordered_set<size_t> get_user_input_match_indices() {
    return user_input_match_indices_;
  }

 protected:
  // Stream on which capture began
  c10::hpu::HPUStream capture_stream_;
  std::recursive_mutex mutex_;
  bool dynamic_env_ = false;
  bool capturing_ = false;
  std::vector<std::shared_ptr<SingleHPUGraph>> captured_graphs;
  std::vector<std::vector<int64_t>> user_input_sizes_ = {};
  std::unordered_set<size_t> user_input_match_indices_;
  // tensors that are input as well as intermediate outputs
  std::vector<habana_lazy::HbLazyTensor> hblazy_tensors_in_out_;
};

} // namespace hpu
} // namespace at
