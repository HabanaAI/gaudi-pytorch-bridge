/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/graph.h"

#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <perf_lib_layer_params.h>
#include <synapse.h>
#include <synapse_api.h>
#include <sys/stat.h>
#include <cstdlib>

#include <algorithm>
#include <ostream>
#include <type_traits>

#include "absl/memory/memory.h"
#include "habana_helpers/logging.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/devmem_logger.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/util.h"
#include "util/time_measure.h"

namespace synapse_helpers {

namespace {

const std::string graph_prefix = ".graph_dumps/";
std::string get_unique_recipe_name(const std::string& name) {
  static uint64_t suffix = -1;

  char* env_graph_prefix{getenv("HBN_TF_GRAPH_PREFIX")};
  if (env_graph_prefix != nullptr) {
    return absl::StrFormat(
        "%s%s_%s_%d", graph_prefix, env_graph_prefix, name, ++suffix);
  }

  return absl::StrFormat("%s%s_%d", graph_prefix, name, ++suffix);
}

bool check_and_prepare_graph_dir() {
  if (mkdir(graph_prefix.c_str(), S_IRWXU | S_IRWXG) ==
      0) { // NOLINT(hicpp-signed-bitwise)
    return true;
  }

  struct stat info {};
  if (stat(graph_prefix.c_str(), &info) != 0 ||
      !(info.st_mode & S_IFDIR)) { // NOLINT(hicpp-signed-bitwise))
    PT_SYNHELPER_WARN("Cannot create graph dump directory ", graph_prefix);
    return false;
  }
  return true;
}

class GraphDirStaticMaker {
 public:
  GraphDirStaticMaker() {
    check_and_prepare_graph_dir();
  }
};

GraphDirStaticMaker graph_dir_maker;

} // namespace

graph::graph(device& device, std::string name)
    : device_{device},
      name_{std::move(name)},
      is_valid_{false},
      in_build_phase_{true},
      in_execution_phase_{false},
      graph_handle_{new synGraphHandle()} {
  static_assert(
      sizeof(ns_ConstantKernel::Params) == 4,
      "ns_ConstantKernel::Params has wrong size");
  static_assert(
      sizeof(ns_Reduction::Params) == 4, "ns_Reduction::Params has wrong size");
  static_assert(
      sizeof(ns_SpatialReduction::Params) == 44,
      "ns_SpatialReduction::Params has wrong size");
  static_assert(
      sizeof(ns_PadKernel::Params) == 44,
      "ns_PadKernel::Params has wrong size");
  static_assert(
      sizeof(ns_TileKernel::Params) == 16,
      "ns_TileKernel::Params has wrong size");
  static_assert(
      sizeof(ns_BatchNormKernel::Params) == 12,
      "ns_BatchNormKernel::Params has wrong size");
  static_assert(
      sizeof(ns_Softmax::Params) == 4, "ns_Softmax::Params has wrong size");
  static_assert(
      sizeof(ns_SoftmaxCrossEntropy::Params) == 8,
      "ns_SoftmaxCrossEntropy::Params has wrong size");
}

std::mutex graph::instance_lock_{};

synapse_error_v<graph> graph::create(device& device, std::string name) {
  graph syn_graph(device, std::move(name));

  if (!(std::getenv("PT_HPU_LAZY_LOWERING")) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    // Lazy mode shape inference call, early return without execution
    return {std::move(syn_graph)};
  }

  PT_SYNHELPER_DEBUG("Graph Create.");
  graph::instance_lock_.lock();
  auto status =
      synGraphCreate(syn_graph.graph_handle_.get(), syn_graph.device_.type());
  SYNAPSE_SUCCESS_CHECK_WITH_OP(
      "Graph creation failed.", status, graph::instance_lock_.unlock())
  syn_graph.is_valid_ = true;
  return {std::move(syn_graph)};
}

graph::graph(graph&& other) noexcept
    : device_{other.device_},
      name_{other.name_},
      is_valid_{other.is_valid_},
      in_build_phase_(other.in_build_phase_),
      in_execution_phase_(other.in_execution_phase_),
      graph_handle_{std::move(other.graph_handle_)} // namespace synapse_helpers
{
  other.is_valid_ = false;
}

graph::~graph() {
  if (is_valid_) {
    PT_SYNHELPER_DEBUG("Graph destroy.");
    synGraphDestroy(*graph_handle_);
    graph::instance_lock_.unlock();

    is_valid_ = false;
  }
}

template <typename T, typename Alloc, template <typename, typename> class V>
std::ostream& operator<<(std::ostream& out, const V<T, Alloc>& collection) {
  auto item{collection.begin()};
  if (item == collection.end()) {
    return out << "<none>";
  }

  out << *item;
  while (++item != collection.end()) {
    out << ", " << reinterpret_cast<void*>(*item);
  }
  return out;
}

synapse_error_o graph::add_node(
    std::vector<synTensor>&& inputs,
    std::vector<synTensor>&& outputs,
    void* const params,
    const unsigned params_size,
    std::string&& node_type) {
  if (!(std::getenv("PT_HPU_LAZY_LOWERING")) &&
      std::getenv("PT_HPU_LAZY_MODE")) {
    // Lazy mode shape inference call, early return without execution
    return {};
  }

  if (!in_build_phase_) {
    return synapse_error{"Graph not in build phase.", synStatus::synFail};
  }

  PT_SYNHELPER_DEBUG(
      "graph ",
      name_,
      " synGenericNodeCreate(inputs=(",
      (&inputs[0]),
      "), outputs=(",
      (&outputs[0]),
      "), sizeInputs=",
      inputs.size(),
      ", sizeOutputs=",
      outputs.size(),
      ", userParams=",
      params,
      ", params_size=",
      params_size,
      ", guid=",
      node_type.c_str(),
      ", name=\"\");");

  // PT always uses node creation with Id
  if (1) {
    synNodeId nodeId;
    auto status = synNodeCreateWithId(
        *graph_handle_,
        inputs.empty() ? nullptr : inputs.data(),
        outputs.empty() ? nullptr : outputs.data(),
        inputs.size(),
        outputs.size(),
        params,
        params_size,
        node_type.c_str(),
        "",
        &nodeId,
        nullptr,
        nullptr);
    if (status != synStatus::synSuccess) {
      PT_SYNHELPER_WARN("Node " + node_type + " add failed.", " Err: ", status);
    }
    HABANA_ASSERT(status == synStatus::synSuccess)
    graph_is_empty_ = false;
    op_to_node_container_pt_["jit_node"].emplace_back(nodeId);
  } else {
    auto status = synNodeCreate(
        *graph_handle_,
        inputs.empty() ? nullptr : inputs.data(),
        outputs.empty() ? nullptr : outputs.data(),
        inputs.size(),
        outputs.size(),
        params,
        params_size,
        node_type.c_str(),
        "",
        nullptr,
        nullptr);
    if (status != synStatus::synSuccess) {
      PT_SYNHELPER_WARN("Node " + node_type + " add failed.", " Err: ", status);
    }
    HABANA_ASSERT(status == synStatus::synSuccess)
    graph_is_empty_ = false;
  }
  return {};
}

synapse_error_v<std::shared_ptr<graph::recipe_handle>> graph::compile() {
  if (!in_build_phase_) {
    return synapse_error{"Graph not in build phase.", synStatus::synFail};
  }
  in_build_phase_ = false;
  synStatus status;
  if (graph_is_empty_) {
    // Valid case, in some special scenarios Op does not add to graph.
    return {};
  }

  status = set_synapse_control_edges();
  SYNAPSE_SUCCESS_CHECK("Setting node dependencies failed.", status);

  TIME_MEASURE_VARS;
  START_TIME_MEASURE;
  auto recipe_handle{absl::make_unique<graph::recipe_handle>(device_)};

  auto name = get_unique_recipe_name(name_);
  status = synGraphCompile(
      &recipe_handle->syn_recipe_handle_,
      *graph_handle_,
      name.c_str(),
      nullptr);

  SYNAPSE_SUCCESS_CHECK("Graph compile failed.", status);
  END_TIME_MEASURE("Synapse graph compilation took");
  in_execution_phase_ = true;
  recipe_handle->graph_is_empty_ = graph_is_empty_;
  recipe_handle->in_execution_phase_ = true;
  recipe_handle->recipe_name_ = std::move(name);

  return {std::move(recipe_handle)};
}

std::string to_string(const std::vector<synLaunchTensorInfo>& patching_info) {
  return absl::StrJoin(
      patching_info, ",", [](std::string* out, const synLaunchTensorInfo& in) {
        absl::StrAppendFormat(out, "%s:0x%X", in.tensorName, in.pTensorAddress);
      });
}

synapse_error_o graph::create_launch_info(
    launch_info& handle,
    const graph::recipe_handle& recipe_handle) {
  synStatus status;

  status = synWorkspaceGetSize(
      &handle.workspace_buffer_size_, recipe_handle.syn_recipe_handle_);
  SYNAPSE_SUCCESS_CHECK("Getting workspace size failed", status);

  return {};
}

synapse_error_o graph::launch(
    launch_info& handle,
    const graph::recipe_handle& recipe_handle,
    const std::vector<synLaunchTensorInfo>& inputs_and_outputs_info) {
  synStatus status;

  if (recipe_handle.graph_is_empty_) {
    // Valid case, in some special scenarios Op does not add to graph.
    return {};
  }

  if (!recipe_handle.in_execution_phase_) {
    return synapse_error{"Graph not in execution phase.", synStatus::synFail};
  }
  handle.runtime_measure_start_ = std::chrono::steady_clock::now();
  PT_SYNHELPER_DEBUG(
      "in graph::launch, launch handle string:\n",
      absl::StrFormat(
          "------Launch-handle------\n"
          "input_outputs_names={%s}\n"
          "-------------------------",
          to_string(inputs_and_outputs_info)));

  auto table_checker{[&recipe_handle](const synLaunchTensorInfo& info) -> bool {
    if (info.pTensorAddress == 0 || info.tensorName == nullptr) {
      PT_SYNHELPER_WARN(
          recipe_handle.recipe_name_,
          " null address:",
          (info.pTensorAddress == 0),
          " null name:",
          (info.tensorName == nullptr),
          " ",
          ((info.tensorName == nullptr) ? "" : info.tensorName));
      return true;
    }
    return false;
  }};
  PT_SYNHELPER_DEBUG("checking input_output patching table");
  SYNAPSE_RETURN_IF_ERROR(
      std::find_if(
          inputs_and_outputs_info.begin(),
          inputs_and_outputs_info.end(),
          table_checker) == inputs_and_outputs_info.end());
  auto& compute_stream = recipe_handle.device_.get_compute_stream();

  if (GET_ENV_FLAG(PT_HABANA_MEM_LOG_LEVEL) == MEM_LOG_GRAPH_LAUNCH) {
    std::string msg = absl::StrFormat(
        "%s%s", "Before launch of graph", recipe_handle.recipe_name_.c_str());
    synapse_helpers::print_live_allocations(msg.c_str());
  }
  status = synLaunch(
      compute_stream,
      inputs_and_outputs_info.data(),
      inputs_and_outputs_info.size(),
      recipe_handle.device_.get_workspace_buffer(handle.workspace_buffer_size_),
      recipe_handle.syn_recipe_handle_);

  SYNAPSE_SUCCESS_CHECK("synLaunch failed.", status)

  return {};
}

synapse_error_v<std::string> graph::name_suffix_from_type(
    const synDataType type) {
  std::string kernel_suffix{};
  if (type == synDataType::syn_type_float) {
    kernel_suffix = "f32";
  } else if (type == synDataType::syn_type_int8) {
    kernel_suffix = "i8";
  } else if (type == synDataType::syn_type_uint8) {
    kernel_suffix = "u8";
  } else if (type == synDataType::syn_type_int16) {
    kernel_suffix = "i16";
  } else if (type == synDataType::syn_type_int32) {
    kernel_suffix = "i32";
  } else if (type == synDataType::syn_type_bf16) {
    kernel_suffix = "bf16";
  } else {
    return synapse_error{"Unknown type", synStatus::synInvalidArgument};
  }
  return kernel_suffix;
}

graph::recipe_handle::~recipe_handle() {
  if (syn_recipe_handle_ &&
      synRecipeDestroy(syn_recipe_handle_) != synStatus::synSuccess) {
    PT_SYNHELPER_WARN("Failed to destroy recipe!");
  }
}

void graph::collect_dst_synapse_nodes(
    graph::Op2NodeContainer::mapped_type& dst_synapse_node_ids,
    const std::string& dst_node) {
  auto op_to_node_iter = op_to_node_container_.find(dst_node);
  if (op_to_node_iter != op_to_node_container_.end() &&
      !op_to_node_iter->second.empty()) {
    dst_synapse_node_ids.insert(
        begin(op_to_node_iter->second), end(op_to_node_iter->second));
    return;
  }

  auto control_edges_iter = control_edges_container_.find(dst_node);
  auto data_edges_iter = data_edges_container_.find(dst_node);
  if (control_edges_iter != end(control_edges_container_)) {
    for (const auto& chained_node : control_edges_iter->second) {
      collect_dst_synapse_nodes(dst_synapse_node_ids, chained_node);
    }
  }

  if (data_edges_iter != end(data_edges_container_)) {
    for (const auto& chained_node : data_edges_iter->second) {
      collect_dst_synapse_nodes(dst_synapse_node_ids, chained_node);
    }
  }
}

synStatus graph::set_synapse_control_edges() {
  synStatus status = synStatus::synSuccess;
  for (const auto& nodePair : control_edges_container_) {
    PT_SYNHELPER_DEBUG(
        "Starting adding synapse control edges from node ", nodePair.first);

    graph::Op2NodeContainer::mapped_type dst_synapse_node_ids;
    for (const auto& dst_node : nodePair.second) {
      collect_dst_synapse_nodes(dst_synapse_node_ids, dst_node);
    }

    auto op_to_node_iter = op_to_node_container_.find(nodePair.first);
    if (op_to_node_iter == end(op_to_node_container_) ||
        op_to_node_iter->second.empty() || dst_synapse_node_ids.empty()) {
      // Some ops like NoOp do not have underlying synapse nodes - it is handled
      // in collect_dst_synapse_nodes

      PT_SYNHELPER_DEBUG(
          "Ommiting adding synapse control edges from node ", nodePair.first);
      continue;
    }
    const auto& src_synapse_node_ids = op_to_node_iter->second;
    std::vector<synNodeId> src_synapse_node_ids_vector(
        begin(src_synapse_node_ids), end(src_synapse_node_ids));
    std::vector<synNodeId> dst_synapse_node_ids_vector(
        begin(dst_synapse_node_ids), end(dst_synapse_node_ids));

    PT_SYNHELPER_DEBUG(
        "Adding synapse control edges from node ", nodePair.first);
    status = synNodeDependencySet(
        *graph_handle_,
        src_synapse_node_ids_vector.data(),
        dst_synapse_node_ids_vector.data(),
        src_synapse_node_ids_vector.size(),
        dst_synapse_node_ids_vector.size());

    PT_SYNHELPER_DEBUG(
        "Added synapse control edges from node ",
        nodePair.first,
        " src size = ",
        src_synapse_node_ids_vector.size(),
        " dst size = ",
        dst_synapse_node_ids_vector.size());

    if (status != synStatus::synSuccess) {
      break;
    }
  }
  return status;
}

synStatus graph::set_synapse_control_edges_pt(
    std::vector<synNodeId> src_synapse_node_ids_vector,
    std::vector<synNodeId> dst_synapse_node_ids_vector) {
  synStatus status = synStatus::synSuccess;

  status = synNodeDependencySet(
      *graph_handle_,
      src_synapse_node_ids_vector.data(),
      dst_synapse_node_ids_vector.data(),
      src_synapse_node_ids_vector.size(),
      dst_synapse_node_ids_vector.size());

  PT_SYNHELPER_DEBUG(
      "Added synapse control edges from node ",
      " src size = ",
      src_synapse_node_ids_vector.size(),
      " dst size = ",
      dst_synapse_node_ids_vector.size());

  return status;
}

} // namespace synapse_helpers
