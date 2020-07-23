/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include "habana_bridge/kernel/hpu_habana_launch_op_pt.h"
#include "habana_device/HPUCheck.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_bridge/kernel/hpu_habana_meta_op_list.h"
#include "habana_device/tensor_builder.h"
#include "habana_helpers/tensor_utils.h"

using namespace torch::jit;

// static initializations
size_t RecipeValueSpec::count = 0;
size_t HabanaLaunchOpPT::instance_count_ = 0;
//--------------------------------------

RecipeArgumentSpec::RecipeArgumentSpec(bool with_grad,
  at::ArrayRef<torch::jit::IValue> input_refs,
  const std::shared_ptr<torch::jit::Graph> &irgraph)
: cas(with_grad, input_refs), hash_code(cas.hashCode()), opstrs(std::string()){
  std::hash<std::string> str_hash;
  for (auto * node : irgraph->nodes()) {
    std::string s(node->kind().toQualString());
    // Adding delemeters for better readability
    opstrs.append("<"+s);
    if (node->kind() == torch::jit::prim::Constant) {
      std::ostringstream oss;
      oss << *node;
      opstrs.append(":"+oss.str());
    }
    opstrs.append(">");
  }
  hash_code = torch::hash_combine(hash_code, str_hash(opstrs));
}

void adjustSizesforPT(at::Tensor *tensor, bool is_output)
{

  auto sizes = tensor->sizes().vec();
  auto strides = tensor->strides().vec();
  at::IntArrayRef out_pos = {0, 3, 1, 2};
  at::IntArrayRef in_pos = {0, 2, 3, 1};

  at::IntArrayRef new_pos_arr = is_output ? out_pos : in_pos;
  auto new_pos = new_pos_arr.vec();
  std::vector<long int> swapped_sizes = {sizes[new_pos[0]],
                                               sizes[new_pos[1]],
                                               sizes[new_pos[2]],
                                               sizes[new_pos[3]]};
  std::vector<long int> swapped_strides = {strides[new_pos[0]],
                                                 strides[new_pos[1]],
                                                 strides[new_pos[2]],
                                                 strides[new_pos[3]]};

  //*tensor_new = at::alias(*tensor);
  tensor->unsafeGetTensorImpl()->set_sizes_and_strides(
            swapped_sizes, swapped_strides);

}

std::ostream &operator<< (std::ostream &O, const RecipeArgumentSpec &v) {
  O << v.hash_code << '\n';
  return O;
}

TensorInfo::TensorInfo (const IValPtr &ivp, const std::string &sn, const ValPtr &vp) {
  TORCH_CHECK(ivp->isTensor(), "aten tensor is expected");
  {
    std::ostringstream oss;
    oss << "%" << vp->debugName();
    ir_name = oss.str();
  }

  syn_name = sn;

  auto pt_tensor = ivp->toTensor();
  {
    std::ostringstream oss;
    oss << pt_tensor.sizes();
    shape_str = oss.str();
  }

  buffer = pt_tensor.data_ptr();
  numel  = pt_tensor.numel();
  size   = pt_tensor.nbytes();
}

TensorInfo::TensorInfo (const IValPtrShared &ivpsh, const std::string &sn, const ValPtr &vp) {
  TORCH_CHECK(ivpsh->isTensor(), "aten tensor is expected");
  {
    std::ostringstream oss;
    oss << "%" << vp->debugName();
    ir_name = oss.str();
  }

  syn_name = sn;

  auto pt_tensor = ivpsh->toTensor();
  {
    std::ostringstream oss;
    oss << pt_tensor.sizes();
    shape_str = oss.str();
  }

  buffer = pt_tensor.data_ptr();
  numel  = pt_tensor.numel();
  size   = pt_tensor.nbytes();
}

std::ostream& operator<< (std::ostream &O, const TensorInfo &t) {
  O << '<'
    << t.ir_name << ':'
    << t.shape_str << ':'
    << t.numel << ':'
    << '(' << t.size << " b)"
    << " :: "
    << t.syn_name  << ':'
    << t.buffer
    << '>';

  return O;
}

std::ostream& operator<< (std::ostream &O, const RecipeValueSpec &v) {
  O << "recipe details ::"
    << " <id : "        << v.id << "> "
    << " <iteration : " << v.iter_idx << "> "
    << " <addr : "      << v.recipe.get() << "> "
    << " <use_count : " << v.recipe.use_count() << "> "
    << '\n';

  if (v.aten_outputs) {
    O << '\n';
    O << "aten_outputs ::";
    for (auto &a : *v.aten_outputs) {
      O << " <dim : " << a->toTensor().dim() << " : " << a->toTensor().sizes() << '>';
    }
  }

  if (v.pinput_indices) {
    O << '\n';
    O << "pinput_indices ::";
    size_t j {0};
    for (auto &a : *v.pinput_indices) {
      O << '\n' << j << " : <";
      for (auto &i : a) {
        O << ' ' << i;
      }
      O << " >";
    }
  }

  if (v.dtensorinfos) {
    O << '\n';
    O << "dtensorinfos ::";
    for (auto &a : *v.dtensorinfos) {
      O << '\n' << "    " << a;
    }
  }

  return O;
}

void RecipeValueSpec::print_hbuff(size_t buf_idx, std::ofstream &out, size_t iteration_count, int numel) {
  float *wb = reinterpret_cast<float *>(htensor_wbuffers->at(buf_idx));
  unsigned buf_size = dtensorinfos->at(buf_idx).size;

  out << "iteration " << iteration_count << " : <"
      << ((buf_idx >= num_inputs) ? "output" : "input") << "> :: < "
      << dtensorinfos->at(buf_idx).ir_name << " : "
      << "shape " << dtensorinfos->at(buf_idx).shape_str << " : "
      << "numel " << dtensorinfos->at(buf_idx).numel << " : "
      << "size (" << buf_size << " b) >";
  out << "<buffer" << '[' << buf_idx << ']' << "@" << dtensorinfos->at(buf_idx).buffer << ">";

  const unsigned max_numel = buf_size / sizeof(float);
  unsigned lim { max_numel };
  if (numel >= 0 ) {
    lim = std::min(lim, (unsigned)numel);
  }

  size_t line_items_num = 8;
  size_t j = 0;
  for (j = 0; j < lim; j++) {
    out << (j%line_items_num ? ' ' : '\n')
        << std::showpoint << std::setw(10) << std::fixed << std::right << wb[j];
  }

  if (lim && lim < max_numel)
    out << (j%line_items_num ? ' ' : '\n') << "...";

  out << '\n';
  if (lim > 0) {
    out << "--------------------" << '\n';
  }
}

void RecipeValueSpec::d2h_dbuff(size_t buf_idx) {
  TORCH_CHECK(num_tensors > buf_idx, "buf_idx is out of range");
  TORCH_CHECK(num_tensors == htensor_wbuffers->size(), "dbuffs hbuffs size mismatch");

  synStatus status;
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();

  unsigned buf_size = dtensorinfos->at(buf_idx).size;

  // allocate the buffer if it is not already allocated
  if (!htensor_wbuffers->at(buf_idx)) {
    status = synHostMalloc(device_id, buf_size, 0, (void **)&(htensor_wbuffers->at(buf_idx)));
    TORCH_CHECK(status == synSuccess, "host-malloc failed");
  }

  synEventHandle upldEvntDone;
  synStreamHandle upStrmHdl = device.get_device_to_host_stream();

  status = synEventCreate(&upldEvntDone, device_id, 0);
  TORCH_CHECK(status == synSuccess, "create upldEvntDone failed");

  status = synMemCopyAsync(upStrmHdl, (uint64_t)dtensorinfos->at(buf_idx).buffer, buf_size,
                           htensor_wbuffers->at(buf_idx), DRAM_TO_HOST);
  TORCH_CHECK(status == synSuccess, "synMemCopyAsync failed");

  status = synEventRecord(upldEvntDone, upStrmHdl);
  TORCH_CHECK(status == synSuccess, "register to signal on d2h copy done");

  status = synStreamSynchronize(upStrmHdl);
  TORCH_CHECK(status == synSuccess, "wait on completion of d2h copy");
}

std::ostream &operator<< (std::ostream &O, const RecipeCacheSimple &v) {
  std::cout << "number of recipes : " << v.map_.size() <<'\n';
  for (auto & i: v.map_) {
    O << "-------------------" << '\n';
    O << "key :: " << *(i.first);
    O << "-------------------" << '\n';
    O << "val :: " << i.second;
    O << "-------------------" << '\n';
  }
  return O;
}

HabanaLaunchOpPT::HabanaLaunchOpPT(const torch::jit::Node* node, bool debug) {
  instance_count_++;
  subgraph_ = node->g(attr::Subgraph);
  opname_ = node->kind().toQualString();
  std::replace(opname_.begin(), opname_.end(), ':', '_');
  debug_ = debug;
  std::ostringstream oss;
  oss << opname_ << '_' << instance_count_;
  id_str = oss.str();
  pt_input_layout = habana::LayoutFormat::NCHW;

  PT_BRIDGE_DEBUG("Creating : ", id_str);

  tensor_dump_numel_ = -2;

  char *snumel = getenv("HABANA_PGM_DUMP_TENSOR_NUMEL");
  if (snumel != nullptr) {
    tensor_dump_numel_ = atoi(snumel);
  }

  enable_tensor_dump_ = (tensor_dump_numel_ >= -1) ? true : false;

  if (enable_tensor_dump_) {
    struct stat st = {0};
    std::string dir_name {"./tensor_dumps"};
    mode_t dir_mode {0755};

    if (stat(dir_name.c_str(), &st) == -1) {
      auto ret = mkdir(dir_name.c_str(), dir_mode);
      TORCH_CHECK(0 == ret, std::string("failed to create " + dir_name));
    }

    dir_name += std::string("/") + id_str;

    if (stat(dir_name.c_str(), &st) == -1) {
      auto ret = mkdir(dir_name.c_str(), dir_mode);
      TORCH_CHECK(0 == ret, std::string("failed to create " + dir_name));
    }

    tdmp_dir_name_ = dir_name;

    {
      std::ostringstream oss;
      oss << tdmp_dir_name_ << "/" << (enable_caching_ ? "tensors_chon" : "tensors_choff") << "_pre.tdmp";
      tdmp_file_name_pre_  = oss.str();

      std::ofstream tensor_file;
      tensor_file.open(tdmp_file_name_pre_.c_str());
      tensor_file << "---- id_str : " << id_str << '\n'
                  << "---- tensor dump of the following graph" << '\n';
      tensor_file << subgraph_->toString()
                  << "----" << '\n' << '\n';
      tensor_file.close();
    }

    {
      std::ostringstream oss;
      oss << tdmp_dir_name_ << "/" << (enable_caching_ ? "tensors_chon" : "tensors_choff") << ".tdmp";
      tdmp_file_name_  = oss.str();

      std::ofstream tensor_file;
      tensor_file.open(tdmp_file_name_.c_str());
      tensor_file << "---- id_str : " << id_str << '\n'
                  << "---- tensor dump of the following graph" << '\n';
      tensor_file << subgraph_->toString()
                  << "----" << '\n' << '\n';
      tensor_file.close();
    }
  }
}

HabanaLaunchOpPT::~HabanaLaunchOpPT() {
  PT_BRIDGE_DEBUG("Destroying : ", id_str);
}

habana::LayoutFormat getPTTensorLayout(at::Tensor& tensor) {

  auto mem_format = tensor.suggest_memory_format();
  if(mem_format == at::MemoryFormat::ChannelsLast ||
     mem_format == at::MemoryFormat::ChannelsLast3d)
      return habana::LayoutFormat::NHWC;
  else
      return habana::LayoutFormat::NCHW;

}

habana::LayoutFormat HabanaLaunchOpPT::getTensorChannelOrder(torch::jit::Value* val) {
  // The value of the node keeps the tensor physical layout memorized
  // We can update this later if we see any changes to the way layouts are handled
  TORCH_CHECK(value_to_tensor_layout.find(val) != std::end(value_to_tensor_layout),
              "HabanaFusion : Channel order not updated");
  return value_to_tensor_layout[val];
}

// See if we are in any leagally accepted channel orders
bool HabanaLaunchOpPT::isChannelOrderSupported(
    torch::jit::Value* val,
    const habana::LayoutFormat &supported_channel_order) {
  return (supported_channel_order == habana::LayoutFormat::ANY)
      || (supported_channel_order == getTensorChannelOrder(val));
}

bool HabanaLaunchOpPT::isInGraphOutputs(torch::jit::Value* value) {
  auto graph_outs = subgraph_->outputs();
  for (auto value_out : graph_outs) {
    if (value->unique() == value_out->unique()) {
      return true;
    }
  }
  return false;
}

void HabanaLaunchOpPT::GetSynapseInputs(
    const HabanaOperatorPtr &habana_op,
    torch::jit::Node* node) {
  auto node_ins = node->inputs();
  for (const auto value_in : node_ins) {
    if (value_to_ivalue[value_in] && value_to_ivalue[value_in]->isTensor()) {
      auto pt_tensor = value_to_ivalue[value_in]->toTensor();

      // Find if an input tensor is already mapped
      // NB: It seems Habana doesn't support shared input to
      // different nodes in graph
      auto is_already_mapped = pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
          std::end(pt_to_synapse_tensors);

      if (is_already_mapped) {
        auto syn_tensor_input = pt_to_synapse_tensors.find(value_to_ivalue[value_in]);
        auto &syn_tensor = habana_op->SetSynapseInput(std::move(syn_tensor_input->second));
        pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], syn_tensor);
      }
      else {
        auto &syn_tensor = habana_op->AllocateSynapseInput(*syn_graph_ptr, &pt_tensor, true);

        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], syn_tensor);

        if (enable_caching_) {
          input_tensorinfo_map.emplace(
              value_to_ivalue[value_in],
              TensorInfo(
                  value_to_ivalue[value_in],
                  syn_tensor.tensor_name_,
                  value_in));
        }
        else {
          input_tensorinfos.emplace_back(
              TensorInfo(
                  value_to_ivalue[value_in],
                  syn_tensor.tensor_name_,
                  value_in));
        }
      }
    }
  }
}

void HabanaLaunchOpPT::GetSynapseOutputs(
    const HabanaOperatorPtr &habana_op,
    torch::jit::Node* node) {
    auto output_tensors_pt = habana_op->GetOutputs();
    auto &output_tensors_syn = habana_op->GetSynOutputs();
    auto &excluded_out_indices = habana_op->GetSynOutputIndicesExcludedInNode();

    auto output_nodes = node->outputs();
    auto habana_kernel_meta_data = habana_op->GetKernelMetaData();
    habana::LayoutFormat out_layout;

    /* Note the input layout information for the node to pass on to output edge */
    auto node_ins = node->inputs();
    habana::LayoutFormat assigned_input_layout = habana::LayoutFormat::NCHW;

    for(auto value_in : node_ins) {
      if (value_to_ivalue[value_in] && value_in->type()->kind() == c10::TypeKind::TensorType) {
        /* Get the input tensor layout information */
        assigned_input_layout = getTensorChannelOrder(value_in);
        break;
      }
    }

    size_t output_nodes_idx = 0, output_tensor_idx = 0;
    TORCH_CHECK(output_nodes.size() == output_tensors_pt.size() - excluded_out_indices.size(),
                "HabanaFusionOp Lowering: Number of output nodes generated doesnt match the graph");

    size_t meta_size = habana_kernel_meta_data.output_layout.size();
    for (synapse_helpers::tensor &out_tensor_syn : output_tensors_syn) {
      out_layout = output_tensor_idx >= meta_size ? habana::LayoutFormat::ANY :
                              habana_kernel_meta_data.output_layout.at(output_tensor_idx);

      /* Pass down the layout information from input to output for layout agnostic
         output (only for single input ans single output op nodes) */
      value_to_tensor_layout[output_nodes[output_nodes_idx]]
        = out_layout == habana::LayoutFormat::ANY ? assigned_input_layout : out_layout;

      if (excluded_out_indices.find(output_tensor_idx) == excluded_out_indices.end()) {
        IValPtrShared ivpsh = std::make_shared<IVal>(output_tensors_pt[output_tensor_idx]);
        value_to_ivalue[output_nodes[output_nodes_idx]] = ivpsh;

        pt_to_synapse_tensors.emplace(value_to_ivalue[output_nodes[output_nodes_idx]], out_tensor_syn);

        output_tensorinfos.emplace_back(
           TensorInfo(
               ivpsh,
               out_tensor_syn.tensor_name_,
               output_nodes[output_nodes_idx]));

        output_nodes_idx++;
      }
      output_tensor_idx++;
  }
}


at::IntArrayRef getDimsForLayout(habana::LayoutFormat channel_order, habana::LayoutFormat current_order) {
  at::IntArrayRef dims;

  if(current_order == habana::LayoutFormat::NCHW)
  {
    if(channel_order == habana::LayoutFormat::NHWC) {
      dims = {0, 2, 3, 1};
    } else if(channel_order == habana::LayoutFormat::HWCK) {
      dims = {2, 3, 1, 0};
    } else {
      TORCH_CHECK(0, " Habana Fusion op permute called for unsupported channel order");
    }
  }
  else if(current_order == habana::LayoutFormat::NHWC)
  {
    if(channel_order == habana::LayoutFormat::NCHW) {
      dims = {0, 3, 1, 2};
    } else if(channel_order == habana::LayoutFormat::HWCK) {
      dims = {1, 2, 3, 0};
    } else {
      TORCH_CHECK(0, " Habana Fusion op permute called for unsupported channel order");
    }
  }
  else if (current_order == habana::LayoutFormat::HWCK) {
    if(channel_order == habana::LayoutFormat::NCHW) {
      dims = {3, 2, 0, 1};
    } else if(channel_order == habana::LayoutFormat::NHWC) {
      dims = {3, 0, 1, 2};
    } else {
      TORCH_CHECK(0, " Habana Fusion op permute called for unsupported channel order");
    }
  }
  else
  {
    TORCH_CHECK(0, " Habana Fusion op permute called for unsupported channel order");
  }


  return dims;
}

// For now, we permute tensors at graph leaves once
// THis function permutes a given tensor to desired layout and modifies
// input_tensor list to have the new tensor

at::Tensor HabanaLaunchOpPT::permuteTensor(
    torch::jit::Value* value_in,
    const at::Tensor &input,
    habana::LayoutFormat permute_order) {

  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  HabanaOperatorPtr permute_kernel = habana::CreateHabanaOperator(
      device_id, "aten::permute", input.scalar_type());
  TORCH_CHECK(
      permute_kernel != nullptr,
      " \n Permute kernel isnt supported in graph mode ");

  habana_kernels.push_back(permute_kernel);
  //set input synapse tensors
  auto is_already_mapped = pt_to_synapse_tensors.find(value_to_ivalue[value_in]) !=
      std::end(pt_to_synapse_tensors);

  if (is_already_mapped) {
    auto syn_tensor_input = pt_to_synapse_tensors.find(value_to_ivalue[value_in]);
    auto &syn_tensor = permute_kernel->SetSynapseInput(std::move(syn_tensor_input->second));
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], syn_tensor);
  } else {
    auto pt_tensor = value_to_ivalue[value_in]->toTensor();
    auto &syn_tensor = permute_kernel->AllocateSynapseInput(
        *syn_graph_ptr, &pt_tensor, true);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], syn_tensor);

    if (enable_caching_) {
      input_tensorinfo_map.emplace(
          value_to_ivalue[value_in],
          TensorInfo(
              value_to_ivalue[value_in],
              syn_tensor.tensor_name_,
              value_in));
    }
    else {
      input_tensorinfos.emplace_back(
          TensorInfo(
              value_to_ivalue[value_in],
              syn_tensor.tensor_name_,
              value_in));
    }
  }

  auto dims = getDimsForLayout(permute_order, value_to_tensor_layout[value_in]);

  torch::jit::Stack input_stack = {IValue(input), IValue(dims)};
  // setup the config params for the kernels
  bool persistent = isInGraphOutputs(value_in);
  permute_kernel->AllocateAndAddSynapseNode(*syn_graph_ptr, input_stack, persistent);
  auto outputs_permute = permute_kernel->GetOutputs();

  //set output synapse tensor
  auto& output_tensors_syn = permute_kernel->GetSynOutputs();
  for (synapse_helpers::tensor &out_tensor_syn : output_tensors_syn) {
    // make the output of permute the input for next synapse kernel
    // permute has a single output
    value_to_ivalue[value_in] = std::make_shared<IVal>(outputs_permute[0]);
    value_to_tensor_layout[value_in] = permute_order;
    pt_to_synapse_tensors.erase(value_to_ivalue[value_in]);
    pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], out_tensor_syn);

    if (persistent) {
      output_tensorinfos.emplace_back(
         TensorInfo(
             value_to_ivalue[value_in],
             out_tensor_syn.tensor_name_,
             value_in));
    }
  }
  return outputs_permute[0];
}

bool HabanaLaunchOpPT::isInGraphInputs(torch::jit::Value* value) {
  auto graph_ins = subgraph_->inputs();
  for (auto value_in : graph_ins) {
    if (value->unique() == value_in->unique()) {
      return true;
    }
  }
  return false;
}

void adjustInputWeight(at::Tensor* tensor)
{
  auto sizes = tensor->sizes().vec();
  auto strides = tensor->strides().vec();
  //TODO : Remove these hardcoded dims, maybe take it from config file?
  at::IntArrayRef new_pos_arr = {2, 3, 1, 0};
  auto new_pos = new_pos_arr.vec();
  std::vector<long int> swapped_sizes = {sizes[new_pos[0]],
                                               sizes[new_pos[1]],
                                               sizes[new_pos[2]],
                                               sizes[new_pos[3]]};
  std::vector<long int> swapped_strides = {strides[new_pos[0]],
                                                 strides[new_pos[1]],
                                                 strides[new_pos[2]],
                                                 strides[new_pos[3]]};
  tensor->unsafeGetTensorImpl()->set_sizes_and_strides(
            swapped_sizes, swapped_strides);

}
void HabanaLaunchOpPT::processInputs(
    torch::jit::Node* node,
    const HabanaOperatorPtr &habana_kernel) {
  //Get the metadata for all inputs, used for preprocessing inputs
  auto &habana_kernel_meta_data = habana_kernel->GetKernelMetaData();
  // Check if its ok to change teh input tensor in the graph attached to value
  auto node_ins = node->inputs();
  size_t tensor_idx = 0;
  habana::LayoutFormat in_layout, prev_layout = habana::LayoutFormat::ANY;
  size_t meta_size = habana_kernel_meta_data.input_layout.size();
  for (const auto value_in : node_ins) {
    if(value_to_ivalue[value_in] && value_in->type()->kind() == c10::TypeKind::TensorType)
    {

        in_layout = tensor_idx >= meta_size ? habana::LayoutFormat::ANY :
                              habana_kernel_meta_data.input_layout.at(tensor_idx);

        if(in_layout == habana::LayoutFormat::ANY && tensor_idx > 0)
        {
          //ATTENTION : We will support only homogeneous layouts for kernels which dont pass meta data requirements for inputs
          // We make inputs homogeneous layouts in case kernel doesnt specify any layout
          //TODO : Add a debug log heres
          in_layout = prev_layout;
        }

        auto tensor = value_to_ivalue[value_in]->toTensor();

        //when we have channel last input, the scripts permute the weights already, so we just change size
        //This should be changed to make it consistent, but requires wider change in eager mode kernels too
        // TODO : Solve this the right way
        if(in_layout == habana::LayoutFormat::HWCK && prev_layout == habana::LayoutFormat::NHWC
            && pt_input_layout == habana::LayoutFormat::NHWC)
        {
          in_layout = habana::LayoutFormat::ANY;
          adjustInputWeight(&tensor);
        }

        if (!(isChannelOrderSupported(value_in, in_layout))) {
          //We only support 4D tensors
          TORCH_CHECK(tensor.dim() <= 4, "WARNING: permute for tensors with dim higher than 4D is not supproted");
          if (tensor.dim() == 4) {
            //permute
            permuteTensor(
                  value_in,
                  tensor,
                  in_layout);
          }
        }
        prev_layout = tensor_idx == 0 ?
                      getTensorChannelOrder(value_in):
                      prev_layout;
        tensor_idx++;
    }
    // TODO : add checks for doing flattening/slicing anything that is
    // required.
  }
}

void HabanaLaunchOpPT::postProcessOutputs() {
  // Do we need a optimization pass here? What should we look for?
  for (auto node : subgraph_->nodes()) {
    auto node_outs = node->outputs();
    for (const auto value_out : node_outs) {
      IValPtrShared ival = value_to_ivalue[value_out];
      if(!ival)
        continue;
      if(!(ival->isTensor()))
        continue;

      if (ival && value_out->type()->kind() == c10::TypeKind::TensorType &&
          isInGraphOutputs(value_out)) {

          auto tensor = ival->toTensor();
          if(getTensorChannelOrder(value_out) != pt_input_layout)
          {
              permuteTensor(
                value_out,
                tensor,
                pt_input_layout);
              if(pt_input_layout == habana::LayoutFormat::NHWC)
              {
                //Make the shape according to NCHW again as PT maintains that even for NHWC tensors
                //Whereas we process internally as NHWC shape only
                adjustSizesforPT(&tensor, true);
                value_to_ivalue.erase(value_out);
                value_to_ivalue[value_out] = std::make_shared<IVal>(tensor);
              }
          }
          else
          {
              if(getTensorChannelOrder(value_out) == habana::LayoutFormat::NHWC)
              {
                //Make the shape according to NCHW again as PT maintains that even for NHWC tensors
                //Whereas we process internally as NHWC shape only
                adjustSizesforPT(&tensor, true);
                value_to_ivalue.erase(value_out);
                value_to_ivalue[value_out] = std::make_shared<IVal>(tensor);
              }

          }

      }
      // TODO : add checks for doing flattening/slicing anything that is
      // required.
    }
  }
}

void HabanaLaunchOpPT::handlePrimNodes(torch::jit::Node* node)
{
  TORCH_CHECK(node->kind() == torch::jit::prim::Constant,
              "Habana Fusion only supports constant type prim nodes");
  auto node_vals = node->outputs();
  for (const auto value : node_vals) {
    IValPtrShared ivptrsh = std::make_shared<IVal>(toIValue(value).value());
    if (ivptrsh->isNone()) {
      continue;
    }
    value_to_ivalue[value] = ivptrsh;
  }
}

torch::jit::Stack HabanaLaunchOpPT::getStackForNode(torch::jit::Node* node) {
  torch::jit::Stack stack_in;
  auto node_inputs = node->inputs();
  for (auto input : node_inputs) {
    if(value_to_ivalue[input])
        stack_in.insert(stack_in.end(), *value_to_ivalue[input]);
    else
        stack_in.insert(stack_in.end(), IValue());
  }
  return stack_in;
}

c10::ScalarType HabanaLaunchOpPT::getNodeScalarType(torch::jit::Node* node) {
  //return the data type of first input tensor
  for (auto input : node->inputs())
    {
      if (value_to_ivalue[input] && value_to_ivalue[input]->isTensor()) {
        return value_to_ivalue[input]->toTensor().scalar_type();
      }
    }
  //Default return float for now if no tensor found
  return c10::ScalarType::Float;
}

void HabanaLaunchOpPT::handleMetaOps(torch::jit::Node* node)
{
  //Call the meta op via CPU impl
  //Some ops dont support c10 op.callBoxed so we need to call via JIT
  torch::jit::Stack stack;
  void *in_data, *out_data;
  auto node_ins = node->inputs();
  habana::LayoutFormat out_layout;
  IValPtrShared input_ptr {nullptr};
  size_t pinput_count {0};

  for (const auto value_in : node_ins) {
    stack.insert(stack.end(), *value_to_ivalue[value_in]);
    if (value_to_ivalue[value_in]->isTensor()) {
      auto tensor = value_to_ivalue[value_in]->toTensor();
      if (pt_to_synapse_tensors.find(value_to_ivalue[value_in]) == std::end(pt_to_synapse_tensors)) {
        in_data = tensor.data_ptr();
        input_ptr = value_to_ivalue[value_in];
        auto dtype =  tensor.scalar_type();
        meta_syn_tensors.push_back(habana_helpers::create_tensor(tensor,
                                  syn_graph_ptr->get_graph_handle(),
                                  true, dtype));
        pt_to_synapse_tensors.emplace(value_to_ivalue[value_in], meta_syn_tensors.back());
        out_layout = value_to_tensor_layout[value_in];

        if (enable_caching_) {
          input_tensorinfo_map.emplace(
              value_to_ivalue[value_in],
              TensorInfo(
                  value_to_ivalue[value_in],
                  meta_syn_tensors.back().tensor_name_,
                  value_in));
        }
        else {
          input_tensorinfos.emplace_back(
              TensorInfo(
                  value_to_ivalue[value_in],
                  meta_syn_tensors.back().tensor_name_,
                  value_in));
        }
        pinput_count++;
      }
    }
  }
  torch::jit::Operator jit_op = node->getOperator();
  auto offset = jit_op.getOperation()(stack);

  auto syn_tensor_input = pt_to_synapse_tensors.find(value_to_ivalue[node_ins[0]]);
  TORCH_CHECK(offset == 0);

  auto node_outs = node->outputs();
  auto outputs = last(stack, node_outs.size());
  int i = 0;
  for (const auto val_out : node_outs) {
    IValPtrShared ival = std::make_shared<IVal>(outputs[i]);
    value_to_ivalue[val_out] = ival;
    if (ival->isTensor()) {
      auto tensor = ival->toTensor();
      value_to_tensor_layout[val_out] = out_layout;
      auto dtype =  tensor.scalar_type();
      //create a tensor variant on the same memory section as the input
      auto variant = synapse_helpers::tensor_builder(
                       tensor.sizes(),
                       habana_helpers::pytorch_to_synapse_type(dtype))
                       .mark_persistence(true)
                       .with_memory_section(syn_tensor_input->second.memorysection())
                       .build(
                       synapse_helpers::HPURegistrar::get_device(
                       tensor.device().index()),
                       syn_tensor_input->second.graph());

      meta_syn_tensors.push_back(absl::get<synapse_helpers::tensor>(std::move(variant)));
      auto &syn_tensor = meta_syn_tensors.back();

      pt_to_synapse_tensors.emplace(value_to_ivalue[val_out], syn_tensor);

      pinput_tensorinfos.emplace_back(
         TensorInfo(
             ival,
             syn_tensor.tensor_name_,
             val_out));

      if (input_to_pinput_indices.end() == input_to_pinput_indices.find(input_ptr)) {
        input_to_pinput_indices.emplace(input_ptr, IdxVec());
      }
      input_to_pinput_indices[input_ptr].push_back(pinput_tensorinfos.size()-1);
      out_data = tensor.data_ptr();
    }
    i++;
  }
  TORCH_CHECK(in_data == out_data, "HabanaFusion : Data pointer changed in Meta op");
}

void HabanaLaunchOpPT::ReorderInputs(RecipeValueSpec &rv) {
  if (enable_caching_) {
    std::vector<IdxVec> pinput_indices;
    // reoroder input_tensorinfos
    bool has_empty_name = false;

    //for (size_t i = pt_stack->size()-num_inputs; i < pt_stack->size(); i++) {
      //torch::jit::IValue *input_ptr = &(pt_stack->at(i));
    for (size_t i = pt_stack_sh.size()-num_inputs; i < pt_stack_sh.size(); i++) {
      IValPtrShared input_ptr = pt_stack_sh.at(i);
      if (input_ptr->isTensor()) {
        auto it = input_tensorinfo_map.find(input_ptr);
        if (it != input_tensorinfo_map.end()) {
          input_tensorinfos.push_back(it->second);
          if (input_to_pinput_indices.end() != input_to_pinput_indices.find(input_ptr)) {
            pinput_indices.push_back(input_to_pinput_indices[input_ptr]);
          }
          else {
            pinput_indices.emplace_back(IdxVec());
          }

          if (it->second.syn_name.empty()) {
            has_empty_name = true;
          }
        }
        else {
          TORCH_CHECK(false, "synapse tensor not found");
        }
      }
    }

    TORCH_CHECK(!has_empty_name, "empty tensor name");
    TORCH_CHECK(input_tensorinfos.size() == num_tensor_inputs, "number of input tensors are not matching");

    rv.pinput_indices = std::make_shared<std::vector<std::vector<size_t>>>(pinput_indices);
  }
}

void HabanaLaunchOpPT::CompileAndExecuteHabanaFusedOpKernel() {
  //PT_BRIDGE_BEGIN;

  // figure out the right device id
  auto& device = synapse_helpers::HPURegistrar::get_device();
  synDeviceId device_id = device.id();
  std::ostringstream oss;
  oss << opname_ << "_" << instance_count_;
  synapse_helpers::graph syn_graph = habana_helpers::create_graph(device_id, oss.str());
  syn_graph_ptr = &syn_graph;

  // for each node in IR graph, at this point the graph is a list with nodes topoloically sorted
  // TODO : check if we need to reorder nodes in any case
  torch::jit::graph_node_list graph_nodes = subgraph_->nodes();
  for (auto* node : graph_nodes) {

    // Prim nodes require special handling and are a special case
    if(node->kind().is_prim()) {
      handlePrimNodes(node);
      continue;
    }

    // If its a meta op we need to call the CPU impl and capture changes
    // Only valid for single tensor ops
    // Can we avoid the string match here?
    if(HabanaMetaOpList::isHabanaMetaOp(node->kind().toQualString())) {
      handleMetaOps(node);
      continue;
    }
    // Get kernel context
    habana::HabanaOperatorPtr HabanaKernel = habana::CreateHabanaOperator(
        device_id, node->kind().toQualString(), getNodeScalarType(node));

    TORCH_CHECK(HabanaKernel != nullptr,
                std::string(" \n  kernel ") +
                std::string(node->kind().toQualString()) +
                std::string(" isnt supported in graph mode "));

    // See if we need to modify/permute tesnors
    processInputs(node, HabanaKernel);

    // Create/attach the synapse inputs from aten tensors
    GetSynapseInputs(HabanaKernel, node);

    torch::jit::Stack input_stack = getStackForNode(node);

    // setup the config params for the kernels
    HabanaKernel->AllocateAndAddSynapseNode(syn_graph, input_stack, true);

    // Get the output tensors created back from the kernel
    // We set type so that the created tensor is propagated throughout graph
    GetSynapseOutputs(HabanaKernel, node);

    //Adding to a vector as we share context through shared pointers and we dont want to
    //call delete untill we are done with whole graph
    habana_kernels.push_back(HabanaKernel);
  }

  postProcessOutputs();

  if (syn_graph.is_empty()) {
    UpdateOutputs();
    return;
  }

  auto&& error_variant {
    syn_graph.compile()
  };
  if (ABSL_PREDICT_FALSE(absl::holds_alternative<synapse_helpers::synapse_error>(error_variant))) {
    auto& error = absl::get<synapse_helpers::synapse_error>(error_variant);
    std::cout << "syn compile encountered : " << error.error << " " << error.status << '\n';
    TORCH_CHECK(false, "syn compile failed");
  }

  auto cur_recipe = get_value(std::move(error_variant));
  RecipeValueSpec rv(cur_recipe);

  ReorderInputs(rv);

  rv.num_inputs = input_tensorinfos.size();

  rv.dtensorinfos = std::make_shared<std::vector<TensorInfo>>(input_tensorinfos);
  rv.dtensorinfos->insert(rv.dtensorinfos->end(), pinput_tensorinfos.begin(), pinput_tensorinfos.end());
  rv.dtensorinfos->insert(rv.dtensorinfos->end(), output_tensorinfos.begin(), output_tensorinfos.end());

  rv.num_tensors = rv.dtensorinfos->size();

  if (enable_tensor_dump_) {
    rv.htensor_wbuffers = std::make_shared<std::vector<uint64_t>>(std::vector<uint64_t>(rv.num_tensors, 0));
  }

  rv.aten_outputs = std::make_shared<std::vector<IValPtrShared>>(std::vector<IValPtrShared>());
  for (auto output : subgraph_->outputs()) {
    rv.aten_outputs->push_back(value_to_ivalue[output]);
  }

  if (enable_tensor_dump_) {
    DumpTensors_pre(rv);
  }

  LaunchRecipe(rv);

  if (enable_tensor_dump_) {
    DumpTensors(rv);
  }

  if (enable_caching_) {
    // Add the <key,value> pair to the map
    std::shared_ptr<RecipeArgumentSpec> ra_spec = std::make_shared<RecipeArgumentSpec>(false, input_refs, subgraph_);
    recipe_cache.add(ra_spec, rv);
  }

  UpdateOutputs();
}

void HabanaLaunchOpPT::DumpTensors_pre(RecipeValueSpec &rv) {
  if (enable_tensor_dump_) {
    std::ofstream tensor_file;
    tensor_file.open(tdmp_file_name_pre_.c_str(), std::ios::out | std::ios::app);
    for (size_t i = 0; i < rv.num_tensors; ++i) {
      rv.d2h_dbuff(i);
      rv.print_hbuff(i, tensor_file, iteration_count_, tensor_dump_numel_);
    }
    tensor_file.close();
  }
}

void HabanaLaunchOpPT::DumpTensors(RecipeValueSpec &rv) {
  if (enable_tensor_dump_) {
    std::ofstream tensor_file;
    tensor_file.open(tdmp_file_name_.c_str(), std::ios::out | std::ios::app);
    for (size_t i = 0; i < rv.num_tensors; ++i) {
      rv.d2h_dbuff(i);
      rv.print_hbuff(i, tensor_file, iteration_count_, tensor_dump_numel_);
    }
    tensor_file.close();
  }
}

bool HabanaLaunchOpPT::CompileSynapseGraph(
    std::shared_ptr<synapse_helpers::graph::recipe_handle>& synh_recipe) {
  auto compile_result = syn_graph_ptr->compile();
  synh_recipe = get_value(std::move(compile_result));
  return (synh_recipe != nullptr);
}

void HabanaLaunchOpPT::LaunchRecipe(RecipeValueSpec &rv) {
  rv.SelfCheck();

  std::vector<synLaunchTensorInfo> syn_launch_info;

  // Populate the <name,buffer> pairs from TensorInfo for synLaunch
  syn_launch_info.reserve(rv.num_tensors);
  for (size_t i = 0; i < rv.num_tensors; ++i)
    syn_launch_info.emplace_back(
        synLaunchTensorInfo{
            rv.dtensorinfos->at(i).syn_name.c_str(),
            reinterpret_cast<uint64_t>(rv.dtensorinfos->at(i).buffer)});

  auto & device = synapse_helpers::HPURegistrar::get_device();

  synStreamHandle stream_handle = device.get_compute_stream();

  synapse_helpers::graph::launch_info ln_info(rv.recipe->device_);
  synapse_helpers::graph::create_launch_info(ln_info, *rv.recipe);

  auto&& error_optional {
    synapse_helpers::graph::launch(ln_info, *rv.recipe, syn_launch_info)
  };
  if (ABSL_PREDICT_FALSE(error_optional.has_value())) {
    auto& error = error_optional.value();
    std::cout << "syn launch encountered : " << error.error << " " << error.status << '\n';
    TORCH_CHECK(false, "syn launch failed");
  }

  TORCH_HABANA_CHECK(synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
}

void HabanaLaunchOpPT::UpdateOutputs() {
  // Update the stack
  drop(*pt_stack, num_inputs);
  for (auto output : subgraph_->outputs()) {
    //pt_stack->insert(pt_stack->end(), *value_to_ivalue[output]);
    if(value_to_ivalue[output])
        pt_stack->insert(pt_stack->end(), *value_to_ivalue[output]);
    else
        pt_stack->insert(pt_stack->end(), IValue());
        //TORCH_CHECK(false, "missing output aten tensor");
  }
}

template<typename T>
void HabanaLaunchOpPT::clearMember(T& m_container)
{
  T empty;
  using std::swap;
  swap(m_container, empty);
}

void HabanaLaunchOpPT::clear() {
  pt_stack = nullptr;
  pt_stack_sh.clear();
  syn_graph_ptr = nullptr;

  habana_kernels.clear();

  input_tensorinfos.clear();
  pinput_tensorinfos.clear();
  output_tensorinfos.clear();
  value_to_tensor_layout.clear();

  value_to_ivalue.clear();
  pt_to_synapse_tensors.clear();
  input_tensorinfo_map.clear();
  meta_syn_tensors.clear();
  input_to_pinput_indices.clear();

  num_tensor_inputs = 0;
}

bool HabanaLaunchOpPT::IsCached(std::shared_ptr<RecipeArgumentSpec> &spec) {
  //Check whether the input signature is changed
  bool is_found(false);
  if (!recipe_cache.empty() && recipe_cache.exists(spec)) {
    is_found = true;
  }
  return is_found;
}

void HabanaLaunchOpPT::run(torch::jit::Stack& stack) {
  PT_BRIDGE_BEGIN;
  num_inputs = subgraph_->inputs().size();
  auto subgraph_inputs = subgraph_->inputs();
  input_refs = last(stack, num_inputs);
  ref_count_++;
  iteration_count_++;


  // Fusion pass should ensure all nodes are on Habana, if all nodes not on habana device, we should assert
  bool is_all_hpu = true;
  for (auto &input : input_refs) {
    if (input.isTensor()) {
      is_all_hpu = input.toTensor().device().type() != c10::DeviceType::HABANA
        ? false
        : is_all_hpu;
    }
  }

  // We dont support running some ops on CPU while running fused op on Habana
  // All tensors should be alocated to habana before entering this phase
  TORCH_CHECK(is_all_hpu == true, " Habana Fusion needs all tensors to be in HPU ");

  // caching :: begin
  if (enable_caching_) {
    std::shared_ptr<RecipeArgumentSpec> spec =
      std::make_shared<RecipeArgumentSpec>(false, input_refs, subgraph_);

    if (IsCached(spec)) {
      // This is cache hit. Run the cached recipe
      RecipeValueSpec rv = recipe_cache.get(spec);

      // Patch the input buffers
      size_t i = 0;

      for (auto const &input : input_refs) {
        if (input.isTensor()) {
          rv.dtensorinfos->at(i).buffer = input.toTensor().data_ptr();
          for (size_t j = 0; j < rv.pinput_indices->at(i).size(); j++) {
            size_t k = rv.num_inputs + (*rv.pinput_indices)[i][j];
            rv.dtensorinfos->at(k).buffer = input.toTensor().data_ptr();
          }
          i++;
        }
      }

      if (enable_tensor_dump_) {
        DumpTensors_pre(rv);
      }

      LaunchRecipe(rv);

      if (enable_tensor_dump_) {
        DumpTensors(rv);
      }

      // Update the stack from the recipe itself
      drop(stack, num_inputs);
      for (const auto & ivptrsh: *(rv.aten_outputs)) {
        stack.insert(stack.end(), *ivptrsh);
      }

      clear();
      PT_BRIDGE_END;
      return;
    }
  }
  // caching :: end
  {
    // Keep a handle to the stack for future use
    pt_stack = &stack;

    size_t j = stack.size()-num_inputs;
    for ( ; j < stack.size(); j++) {
      IValPtrShared ivptrsh = std::make_shared<IVal>(stack[j]);
      pt_stack_sh.push_back(ivptrsh);
    }

    pt_input_layout = habana::LayoutFormat::NCHW;
    for (size_t j = 0; j < pt_stack_sh.size(); j++) {
      auto value_input = subgraph_inputs[j];

      if (pt_stack_sh[j]->isTensor()) {

        // Taking alias as that allows us to detach it from PT and do metadata changes
        // It gives us more control over tensor changes, but caution is needed.
        // Its might be a bit dangerous, but only way to communicate layour changes
        // PT doesnt allow any stride changes we want, we can review it with PT folks

        auto tensor = at::alias(pt_stack_sh[j]->toTensor());

        //Get  the logical layout from PT tensor
        //We dont touch this, even while doing permutes, the PT logical tensor is retained
        //For us all tensors are contiguous
        //PT doesnt let us mark logical layout directly so we dont change them

        value_to_tensor_layout[value_input] = getPTTensorLayout(tensor);
        if (getPTTensorLayout(tensor) == habana::LayoutFormat::NHWC) {
          // Make the sizes according to NCHW as PT maintains
          // NCHW shapes even for NHWC tensors(It doesnt change shape)
          adjustSizesforPT(&tensor, false);
          IValPtrShared ivptrsh = std::make_shared<IVal>(tensor);
          value_to_ivalue[value_input] = ivptrsh;
          pt_stack_sh[j] = ivptrsh;
          pt_input_layout = habana::LayoutFormat::NHWC;
        }
        else {
          value_to_ivalue[value_input] = pt_stack_sh[j];
        }
        num_tensor_inputs++;
      }
      else {
        value_to_ivalue[value_input] = pt_stack_sh[j];
      }
    }
  }

  //<Decription> This is the main function that
  //  a. creates the HabanaLaunchOp
  //  b. compiles and executes the same
  CompileAndExecuteHabanaFusedOpKernel();

  // clear the context
  // TODO : See if we need to add a contect to this object pointer or clearing like this is good?
  clear();

  PT_BRIDGE_END;
}
