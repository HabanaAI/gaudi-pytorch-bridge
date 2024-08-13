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
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/types/optional.h>
#include <absl/types/variant.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <ostream>
#include <type_traits>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"

#include "backend/helpers/eager_pipeline.h"
#include "backend/helpers/event_dispatcher.h"
#include "backend/helpers/runtime_config.h"
#include "backend/synapse_helpers/device.h"
#include "backend/synapse_helpers/devmem_logger.h"
#include "backend/synapse_helpers/env_flags.h"
#include "backend/synapse_helpers/graph.h"
#include "backend/synapse_helpers/stream.h"
#include "backend/synapse_helpers/tensor_builder_base.h"
#include "backend/synapse_helpers/util.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/stat_collection.h"
#include "habana_helpers/towl.h"
#include "habana_lazy/memlog.h"
#include "util/time_measure.h"

namespace synapse_helpers {

namespace {

const std::string graph_prefix = ".graph_dumps/";
std::string get_unique_recipe_name(const std::string& name, bool eager_mode) {
  static uint64_t suffix = -1;
  if (eager_mode && !(IS_SYNHELPER_DEBUG_ENABLED)) {
    return std::to_string(++suffix);
  }

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

std::unordered_map<std::string, std::string> ParseHintsFromString(
    const std::string& hints_str) {
  // parse hints based on form "name1:value1;[name2:value2;]"
  std::vector<std::string> hints_vec;
  std::stringstream ss(hints_str);
  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ';');
    if (substr.empty())
      break;
    hints_vec.push_back(substr);
  }

  std::unordered_map<std::string, std::string> hints_map;
  for (const auto& s : hints_vec) {
    auto pos = s.find(":");
    std::string key(s.substr(0, pos));
    std::string val(s.substr(pos + 1));
    hints_map.emplace(key, val);
  }
  return hints_map;
}

} // namespace

#define CHECK_KPARAMS_SIZE(name, size) \
  static_assert(                       \
      sizeof(name::Params) == size,    \
      #name "::Params size has changed. Update TF code.");

CHECK_KPARAMS_SIZE(ns_ConstantKernel, 4)
CHECK_KPARAMS_SIZE(ns_Reduction, 4)
CHECK_KPARAMS_SIZE(ns_SpatialReduction, 44)
CHECK_KPARAMS_SIZE(ns_PadKernel, 44)
CHECK_KPARAMS_SIZE(ns_TileKernel, 16)
CHECK_KPARAMS_SIZE(ns_BatchNormKernel, 12)
CHECK_KPARAMS_SIZE(ns_Softmax, 4)
CHECK_KPARAMS_SIZE(ns_SoftmaxCrossEntropy, 8)

#undef CHECK_KPARAMS_SIZE

graph::graph(device& device, std::string name)
    : device_{device}, name_{std::move(name)} {}

graph graph::create(
    device& device,
    std::string name,
    bool dry_run,
    bool eager_mode) {
  PT_SYNHELPER_BEGIN;
  graph syn_graph(device, std::move(name));

  syn_graph.dry_run_ = dry_run;
  syn_graph.eager_mode_ = eager_mode;
  if (syn_graph.dry_run_ == false) {
    synStatus status = synSuccess;
    const auto device_type = syn_graph.device_.type();
    if (syn_graph.eager_mode_ && device_type != synDeviceGaudi &&
        GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EAGER_COMPILER)) {
      status = synGraphCreateEager(&syn_graph.graph_handle_, device_type);
    } else {
      status = synGraphCreate(&syn_graph.graph_handle_, device_type);
    }
    HABANA_ASSERT(
        status == synStatus::synSuccess,
        "Graph creation failed. synStatus=",
        Logger::formatStatusMsg(status));

    if (habana_helpers::IsInferenceMode()) {
      synStatus status = synSuccess;
      bool quantizationEnabled = habana_helpers::IsQuantizationEnabled();
      uint64_t values[] = {true, quantizationEnabled};
      synGraphAttribute att[] = {
          GRAPH_ATTRIBUTE_INFERENCE, GRAPH_ATTRIBUTE_QUANTIZATION};
      const uint32_t size = 2;
      status = synGraphSetAttribute(syn_graph.graph_handle_, att, values, size);
      HABANA_ASSERT(
          status == synStatus::synSuccess,
          "Failed to set graph attributes. synStatus=",
          Logger::formatStatusMsg(status));
    }
  }

  syn_graph.is_valid_ = true;
  PT_SYNHELPER_END;
  return syn_graph;
}

graph graph::create_for_refinement(device& device, std::string name) {
  PT_SYNHELPER_BEGIN;
  graph syn_graph(device, std::move(name));

  synStatus status = synSuccess;
  status = synGraphCreate(&syn_graph.graph_handle_, syn_graph.device_.type());
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Graph creation failed. synStatus=",
      Logger::formatStatusMsg(status));
  syn_graph.is_valid_ = true;
  syn_graph.dry_run_ = false;
  PT_SYNHELPER_END;
  return syn_graph;
}

std::pair<std::vector<synTensorHandleMap>, std::vector<synNodeHandleMap>> graph::
    duplicate(synGraphHandle& duplicate_graph_handle) {
  PT_SYNHELPER_BEGIN;

  HABANA_ASSERT(
      eager_mode_,
      "Graph Duplicate API is not supposed to be used in lazy mode");

  if (numTensors == 0 && numNodes == 0) {
    // first duplicate call to obtain number of tensors and nodes in the
    // graph
    auto status = synGraphDuplicate(
        graph_handle_,
        &duplicate_graph_handle,
        nullptr,
        &numTensors,
        nullptr,
        &numNodes);
    HABANA_ASSERT(
        status == synStatus::synSuccess,
        "Graph duplication failed. synStatus=",
        Logger::formatStatusMsg(status));
  }

  std::vector<synTensorHandleMap> tensorsMap(numTensors);
  std::vector<synNodeHandleMap> nodesMap(numNodes);

  auto status = synGraphDuplicate(
      graph_handle_,
      &duplicate_graph_handle,
      tensorsMap.data(),
      &numTensors,
      nodesMap.data(),
      &numNodes);

  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Graph duplication failed. synStatus=",
      Logger::formatStatusMsg(status));
  graph_is_empty_ = false;
  PT_SYNHELPER_END;
  return std::make_pair(tensorsMap, nodesMap);
}

bool graph::inferShapes() {
  PT_SYNHELPER_BEGIN;

  PT_SYNHELPER_DEBUG("Infer Shapes on Duplicate Graph.");
  synStatus status = synSuccess;
  if (eager_mode_) {
    status = synGraphInferShapes(graph_handle_);
  } else {
    HABANA_ASSERT(
        0 && "Infer Shapes API is not supposed to be used in lazy mode");
  }
  PT_SYNHELPER_END;
  return synStatus::synSuccess == status;
}

void graph::getTensorGeometry(
    synTensor tensor_handle,
    std::vector<int64_t>& shape) {
  PT_SYNHELPER_BEGIN;
  synStatus status = synSuccess;
  synTensorGeometry tensorGeometry;
  status =
      synTensorGetGeometry(tensor_handle, &tensorGeometry, synGeometrySizes);
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Tensor Get Geometry failed. synStatus=",
      Logger::formatStatusMsg(status))

  shape.resize(tensorGeometry.dims);
  for (size_t i = 0; i < shape.size(); i++) {
    shape[i] = tensorGeometry.sizes[shape.size() - i - 1];
  }
  PT_SYNHELPER_END;
}

void graph::setTensorGeometry(
    synTensor tensor_handle,
    std::vector<int64_t> shape) {
  PT_SYNHELPER_BEGIN;
  synStatus status = synSuccess;
  synTensorGeometry maxGeometry;
  maxGeometry.dims = shape.size();

  for (size_t i = 0; i < shape.size(); i++) {
    maxGeometry.sizes[shape.size() - i - 1] = shape.at(i);
  }

  status = synTensorSetGeometry(tensor_handle, &maxGeometry, synGeometrySizes);
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Tensor Set Geometry failed. synStatus=",
      Logger::formatStatusMsg(status))
  PT_SYNHELPER_END;
}

void graph::setTensorPermutation(
    synTensor tensor_handle,
    std::vector<uint8_t>& permute_or_empty) {
  PT_SYNHELPER_BEGIN;
  synStatus status = synSuccess;
  synTensorPermutation perm = {};
  perm.dims = permute_or_empty.size();
  for (size_t i = 0; i < perm.dims; i++) {
    perm.permutation[i] = permute_or_empty[i];
  }

  status = synTensorSetPermutation(tensor_handle, &perm);
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Tensor Set Permutation failed. synStatus=",
      Logger::formatStatusMsg(status))
  PT_SYNHELPER_END;
}

void graph::setTensorSectionOffset(synTensor tensor_handle, uint64_t offset) {
  PT_SYNHELPER_BEGIN;
  synStatus status = synSuccess;

  status = synTensorSetSectionOffset(tensor_handle, offset);
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Tensor Set Section Offset failed. synStatus=",
      Logger::formatStatusMsg(status))
  PT_SYNHELPER_END;
}

void graph::getNodeParams(
    const synGraphHandle graph_handle,
    const synNodeId node_id,
    void* node_params,
    unsigned* params_size) {
  PT_SYNHELPER_BEGIN;
  synStatus status = synSuccess;

  status =
      synNodeGetUserParams(graph_handle, node_id, node_params, params_size);
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Get Node Params failed. synStatus=",
      Logger::formatStatusMsg(status))
  PT_SYNHELPER_END;
}

void graph::setNodeParams(
    const synGraphHandle graph_handle,
    const synNodeId node_id,
    const void* node_params,
    const unsigned params_size) {
  PT_SYNHELPER_BEGIN;
  synStatus status = synSuccess;

  status =
      synNodeSetUserParams(graph_handle, node_id, node_params, params_size);
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Set Node Params failed. synStatus=",
      Logger::formatStatusMsg(status))
  PT_SYNHELPER_END;
}

std::
    tuple<graph, std::vector<synTensorHandleMap>, std::vector<synNodeHandleMap>>
    graph::duplicate(graph& other) {
  graph dup_graph(other.device_, other.name_);
  auto [tensors_handle_map, nodes_handle_map] =
      other.duplicate(dup_graph.graph_handle_);
  dup_graph.is_valid_ = other.is_valid_;
  dup_graph.in_build_phase_ = other.in_build_phase_;
  dup_graph.in_execution_phase_ = other.in_execution_phase_;
  dup_graph.dry_run_ = other.dry_run_;
  dup_graph.numInterTensors = other.numInterTensors;
  dup_graph.is_shape_agnostic_graph_ = other.is_shape_agnostic_graph_;
  dup_graph.eager_mode_ = other.eager_mode_;
  dup_graph.dynamic_graph_ = dup_graph.dynamic_graph_;
  dup_graph.enable_optim_output_sif_ = other.enable_optim_output_sif_;
  dup_graph.numTensors = other.numTensors;
  dup_graph.numConstTensors = other.numConstTensors;
  dup_graph.numInterTensors = other.numInterTensors;
  dup_graph.numShapeTensors = other.numShapeTensors;
  dup_graph.numNodes = other.numNodes;
  dup_graph.graph_is_empty_ = other.graph_is_empty_;
  dup_graph.syn_node_id_vec_ = other.syn_node_id_vec_;
  return {
      std::move(dup_graph),
      std::move(tensors_handle_map),
      std::move(nodes_handle_map)};
}

graph::graph(graph&& other) noexcept
    : device_{other.device_},
      name_{other.name_},
      is_valid_{other.is_valid_},
      in_build_phase_(other.in_build_phase_),
      in_execution_phase_(other.in_execution_phase_),
      graph_is_empty_(other.graph_is_empty_),
      graph_handle_(other.graph_handle_),
      dry_run_(other.dry_run_),
      dynamic_graph_(other.dynamic_graph_),
      enable_optim_output_sif_(other.enable_optim_output_sif_),
      numTensors(other.numTensors),
      numConstTensors(other.numConstTensors),
      numInterTensors(other.numInterTensors),
      numShapeTensors(other.numShapeTensors),
      numNodes(other.numNodes),
      is_shape_agnostic_graph_(other.is_shape_agnostic_graph_),
      eager_mode_(other.eager_mode_),
      syn_node_id_vec_(other.syn_node_id_vec_) {
  other.is_valid_ = false;
  other.graph_handle_ = {};
}

graph::~graph() {
  if (is_valid_) {
    PT_SYNHELPER_DEBUG("Graph destroy.");
    if (graph_handle_ != nullptr) {
      synGraphDestroy(graph_handle_);
    }

    is_valid_ = false;
  }
}

template <
    typename T,
    typename Alloc,
    template <typename, typename>
    class V,
    typename std::enable_if<std::negation<typename std::is_same<
        std::string,
        typename V<T, Alloc>::value>::value>::type>::type>
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

void graph::add_node(
    std::vector<synTensor>&& inputs,
    std::vector<synTensor>&& outputs,
    void* const params,
    const unsigned params_size,
    const std::string& node_type,
    synNodeId* ret_node_id,
    const char** input_layouts,
    const char** output_layouts,
    bool deterministic,
    const std::string& hints_str) {
  if (dry_run_) {
    // Lazy mode shape inference call, early return without execution
    return;
  }

  for (auto& tensor : outputs) {
    PT_LAZY_DEBUG(
        "synapse output tensors should not carry permutation. Clearing the permutation from synapse output tensor.");
    synTensorPermutation perm;
    perm.dims = 0;
    auto status = synTensorSetPermutation(tensor, &perm);
    TORCH_HABANA_CHECK(status, "Node " + node_type + "  failed.");
  }
  PT_BRIDGE_DEBUG("\nAdding Node to graph with guid = ", node_type.c_str());
  HABANA_ASSERT(in_build_phase_, "Graph not in build phase.");
  static int cnt = 0;
  std::string node_name;
  if (current_op_name_) {
    node_name +=
        *current_op_name_ + "/" + node_type + "/" + std::to_string(cnt++);
  }
  synapse_helpers::detail::tensor_name_generator::to_netron_syntax(node_name);

  PT_SYNHELPER_DEBUG(
      "graph ",
      name_,
      " synGenericNodeCreate(inputs=(",
      reinterpret_cast<void*>(inputs.empty() ? nullptr : inputs[0]),
      "), outputs=(",
      reinterpret_cast<void*>(outputs.empty() ? nullptr : outputs[0]),
      "), sizeInputs=",
      inputs.size(),
      ", sizeOutputs=",
      outputs.size(),
      ", userParams=",
      params,
      ", params_size=",
      params_size,
      ", guid=",
      node_type.c_str());

  synNodeId nodeId;
  auto status = synNodeCreateWithId(
      graph_handle_,
      inputs.empty() ? nullptr : inputs.data(),
      outputs.empty() ? nullptr : outputs.data(),
      inputs.size(),
      outputs.size(),
      params,
      params_size,
      node_type.c_str(),
      node_name.c_str(),
      &nodeId,
      input_layouts,
      output_layouts);
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "synNodeCreateWithId failed for node: ",
      node_type,
      " with ",
      Logger::formatStatusMsg(status),
      ".");

  graph_is_empty_ = false;
  op_to_node_container_pt_["jit_node"].emplace_back(nodeId);
  if (ret_node_id) {
    *ret_node_id = nodeId;
  }
  if (eager_mode_) {
    syn_node_id_vec_.push_back(nodeId);
    PT_BRIDGE_DEBUG("[SHAPE AGNOSTIC] Adding syn node id: ", nodeId);
  }
  PT_BRIDGE_DEBUG("Adding Syn graph::add_node val ", deterministic);
  if (deterministic) {
    auto status = synNodeSetDeterministic(graph_handle_, nodeId, deterministic);
    if (status != synStatus::synSuccess) {
      PT_SYNHELPER_WARN(
          Logger::formatStatusMsg(status),
          "Node " + node_type + " synNodeSetDeterministic");
      PT_SYNHELPER_FATAL(
          Logger::formatStatusMsg(status),
          "node add synNodeSetDeterministic failed");
    }
  }

  // set hints via synapse API call
  if (!hints_str.empty()) {
    auto hints_map = ParseHintsFromString(hints_str);

    bool is_exec_order_provided = false, is_group_id_provided = false;
    synUserExecOrder user_exec_order;
    synUserProgrammability user_programmability;
    for (const auto& h : hints_map) {
      if (h.first == "exec_order") {
        user_exec_order.executionOrderedIndex = static_cast<unsigned>(std::stoi(h.second));
        is_exec_order_provided = true;
      } else if (h.first == "group_id") {
        user_exec_order.groupId = static_cast<unsigned>(std::stoi(h.second));
        is_group_id_provided = true;
      }
    }

    // require both exec_order and group_id to be set
    if (is_exec_order_provided && is_group_id_provided) {
      user_programmability.userExecOrder = &user_exec_order;
      auto status = synNodeSetUserProgrammability(
          graph_handle_, nodeId, &user_programmability);
      if (status != synStatus::synSuccess) {
        PT_SYNHELPER_WARN(
            Logger::formatStatusMsg(status),
            "Node " + node_type + " synNodeSetUserProgrammability");
        PT_SYNHELPER_FATAL(
            Logger::formatStatusMsg(status),
            "node add synNodeSetUserProgrammability failed");
      }
    } else {
      PT_BRIDGE_DEBUG(
          "Skip calling synNodeSetUserProgrammability due to missing either exec_order or group_id hints for node: ",
          node_type);
    }
  }
}

std::shared_ptr<graph::recipe_handle> graph::compile() {
  HABANA_ASSERT(in_build_phase_, "Graph not in build phase");
  in_build_phase_ = false;
  if (graph_is_empty_) {
    // Valid case, in some special scenarios Op does not add to graph.
    return {};
  }
  STAT_START(synapse_compilation);

  auto status = set_synapse_control_edges();
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Setting node dependencies failed. synStatus=",
      Logger::formatStatusMsg(status));

  TIME_MEASURE_VARS;
  START_TIME_MEASURE;
  auto recipe_handle{absl::make_unique<graph::recipe_handle>()};

  auto name = get_unique_recipe_name(name_, eager_mode_);

  status = synGraphCompile(
      &recipe_handle->syn_recipe_handle_, graph_handle_, name.c_str(), nullptr);

  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Graph compile failed. synStatus=",
      Logger::formatStatusMsg(status));
  END_TIME_MEASURE("Synapse graph compilation took");
  in_execution_phase_ = true;
  recipe_handle->graph_is_empty_ = graph_is_empty_;
  recipe_handle->in_execution_phase_ = true;
  recipe_handle->recipe_name_ = std::move(name);

  auto syn_compile_duration_us =
      std::chrono::duration_cast<std::chrono::duration<int64_t, std::micro>>(
          end_time - start_time)
          .count();
  if (!eager_mode_) {
    habana_helpers::EmitEvent(
        habana_helpers::EventDispatcher::Topic::GRAPH_COMPILE,
        habana_helpers::EventDispatcher::EventParams(
            {{"duration", std::to_string(syn_compile_duration_us)},
             {"recipe", recipe_handle->recipe_name_}}));
  }

  STAT_ADD_ATTRIBUTE(
      globalStatPtsEnum::recipe_compile,
      "Recipe Name",
      recipe_handle->recipe_name_);
  STAT_COLLECT_TIME(synapse_compilation, globalStatPtsEnum::recipe_compile);

  return recipe_handle;
}

std::string to_string(const std::vector<synLaunchTensorInfo>& patching_info) {
  return absl::StrJoin(
      patching_info, ",", [](std::string* out, const synLaunchTensorInfo& in) {
        absl::StrAppendFormat(
            out,
            "%s:%u:0x%X [%s]",
            in.tensorName,
            in.tensorId,
            in.pTensorAddress,
            absl::StrJoin(
                std::begin(in.tensorSize), std::end(in.tensorSize), ","));
      });
}

uint64_t graph::query_workspace_size(
    const graph::recipe_handle& recipe_handle) {
  uint64_t workspace_size;
  auto status =
      synWorkspaceGetSize(&workspace_size, recipe_handle.syn_recipe_handle_);
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Getting workspace size failed. synStatus=",
      Logger::formatStatusMsg(status));
  return workspace_size;
}

void graph::query_recipe_tensor_info(
    const graph::recipe_handle& recipe_handle,
    std::vector<synRetrievedLaunchTensorInfo>& tensor_info_vec) {
  auto status = synTensorRetrieveLaunchInfoById(
      recipe_handle.syn_recipe_handle_,
      tensor_info_vec.size(),
      tensor_info_vec.data());
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "Getting launch tensor info failed, synStatus=",
      Logger::formatStatusMsg(status));
}

void graph::launch(
    device& device,
    const graph::recipe_handle& recipe_handle,
    uint64_t workspace_size,
    std::vector<synLaunchTensorInfo>&& inputs_and_outputs_info,
    std::unique_ptr<device_ptr_lock>& address_lock,
    std::vector<shared_event>& ext_events,
    stream& compute_stream,
    size_t active_graph_key) {
  return launch(
      device,
      recipe_handle,
      workspace_size,
      inputs_and_outputs_info,
      address_lock,
      ext_events,
      compute_stream,
      active_graph_key);
}

void graph::launch(
    device& device,
    const graph::recipe_handle& recipe_handle,
    uint64_t workspace_size,
    std::vector<synLaunchTensorInfo>& inputs_and_outputs_info,
    std::unique_ptr<device_ptr_lock>& address_lock,
    std::vector<shared_event>& ext_events,
    stream& compute_stream,
    size_t active_graph_key) {
  PT_SYNHELPER_BEGIN;
  synStatus status = synStatus::synSuccess;

  if (recipe_handle.graph_is_empty_) {
    // Valid case, in some special scenarios Op does not add to graph.
    return;
  }

  HABANA_ASSERT(
      recipe_handle.in_execution_phase_, "Graph not in execution phase.");

  for (synLaunchTensorInfo& tensorInfo : inputs_and_outputs_info) {
    // [SW-96080], due to change in get_tensor_for_scalar PT tensor has
    // size [0] for 0d tensor need to force it [1] to pass to synapse
    // correctly valdity check for pTensorAddress to differentiate from ZST in
    // case of ZST pTensorAddress will be NULL
    if (tensorInfo.pTensorAddress && tensorInfo.tensorSize[0] == 0) {
      tensorInfo.tensorSize[0] = 1;
    }
  }

  PT_SYNHELPER_DEBUG("STREAM:: Launch recipe with stream::", compute_stream);

  auto table_checker{[&recipe_handle](const synLaunchTensorInfo& info) -> bool {
    if (info.tensorName == nullptr || info.tensorName[0] == '\0') {
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
  HABANA_ASSERT(
      std::find_if(
          inputs_and_outputs_info.begin(),
          inputs_and_outputs_info.end(),
          table_checker) == inputs_and_outputs_info.end(),
      "Checking input_output patching table. failed");

  std::vector<device_ptr> addresses(
      inputs_and_outputs_info.size(), device_nullptr);
  std::unordered_map<uint64_t, uint64_t> host_address_map;
  for (size_t i = 0; i < inputs_and_outputs_info.size(); i++) {
    auto& info = inputs_and_outputs_info[i];
    if (info.tensorType != HOST_TO_DEVICE_TENSOR) {
      addresses[i] = info.pTensorAddress;
    } else {
      host_address_map[i] = info.pTensorAddress;
      addresses[i] = static_cast<uint64_t>(0);
    }
  }

  size_t least_workspace_size = workspace_size;
  size_t tensor_mem =
      device.get_device_memory().get_total_memory_required(addresses);
  if (GET_ENV_FLAG_NEW(PT_ENABLE_WORKSPACE_MEMORY_SHRINK, 1)) {
    bool oom_may = !device.get_device_memory().is_memory_available(tensor_mem);
    if (oom_may) {
      least_workspace_size =
          device.get_least_workspace_size(tensor_mem, workspace_size);
      // Set minimal size of workspace to 4MB to prevent it from being 0
      if (least_workspace_size < 4 * 1024 * 1024) {
        least_workspace_size = 4 * 1024 * 1024;
      }
      device.cleanup_workspace_buffer();
    }
  }
  auto workspace_buffer = device.get_workspace_buffer(least_workspace_size);

  habana_lazy::log_dev_mem_stats(
      "Post-Workspace", recipe_handle.recipe_name_, workspace_size);
  towl::emitDeviceMemorySummary("Post-Workspace");
  if (synapse_helpers::memory_reporter_enable() && active_graph_key > 0) {
    synapse_helpers::MemoryReporter* reporter =
        device.get_device_memory().get_memory_reporter();
    reporter->getGraphStats()->updateGraph(
        active_graph_key, tensor_mem, device.get_workspace_size());
    memory_reporter_event_create(device, MEM_REPORTER_GRAPH_BEFORE_LAUNCH);
  }

  log_graph_info(
      device,
      recipe_handle.recipe_name_.c_str(),
      tensor_mem,
      workspace_size,
      device.get_workspace_size());

  {
    address_lock = absl::make_unique<device_ptr_lock>(
        device.lock_addresses(absl::Span<const device_ptr>(addresses)));
    auto iter = inputs_and_outputs_info.begin();
    size_t index = 0;
    for (auto address : *address_lock) {
      if (GET_ENV_FLAG_NEW(PT_HPU_POOL_MEM_ENABLE_TENSOR_INFO)) {
        log_tensor_info(
            ((iter->tensorName == nullptr) ? "" : iter->tensorName),
            index,
            addresses[index],
            address);
      }

      if (host_address_map.count(index)) {
        iter->pTensorAddress = host_address_map[index];
      } else {
        iter->pTensorAddress = address;
      }
      ++index;
      ++iter;
    }
    towl::emitRecipeLaunch(
        recipe_handle, workspace_size, addresses, inputs_and_outputs_info);

    PT_SYNHELPER_DEBUG(
        "in graph::launch, launch handle string:\n",
        "------Launch-handle ",
        recipe_handle.recipe_name_,
        "------\n"
        "input_outputs_names=",
        to_string(inputs_and_outputs_info),
        "\n"
        "-------------------------",
        "Recipe handle: ",
        fmt::ptr(recipe_handle.syn_recipe_handle_));

    habana_lazy::log_dev_mem_stats(
        "Post-Tensors",
        recipe_handle.recipe_name_,
        device.get_device_memory().get_total_memory_required(addresses));
    towl::emitDeviceMemorySummary("Post-Tensors");

    uint32_t flags{0};
    std::vector<synEventHandle> event_handles;
    event_handles.reserve(ext_events.size());
    std::transform(
        ext_events.begin(),
        ext_events.end(),
        std::back_inserter(event_handles),
        [](shared_event& event) -> synEventHandle { return *event; });

    if (!GET_ENV_FLAG_NEW(PT_COMPILE_ONLY_MODE)) {
      status = synLaunchWithExternalEvents(
          compute_stream,
          inputs_and_outputs_info.data(),
          inputs_and_outputs_info.size(),
          workspace_buffer,
          recipe_handle.syn_recipe_handle_,
          event_handles.data(),
          event_handles.size(),
          flags);
    }
  }

  if (synapse_helpers::memory_reporter_enable() && active_graph_key > 0) {
    synapse_helpers::MemoryReporter* reporter =
        device.get_device_memory().get_memory_reporter();
    reporter->getGraphStats()->updateGraph(
        active_graph_key, tensor_mem, device.get_workspace_size());
    memory_reporter_event_create(device, MEM_REPORTER_GRAPH_AFTER_LAUNCH);
  }
  HABANA_ASSERT(
      status == synStatus::synSuccess,
      "synLaunch failed. synStatus=",
      Logger::formatStatusMsg(status))
  PT_SYNHELPER_END;

  char* sync_launch = getenv("PT_HPU_SYNC_LAUNCH");
  if (sync_launch != nullptr && atoi(sync_launch) == 1) {
    device.synchronize();
  }

  return;
}

std::string_view graph::name_suffix_from_type(
    const synDataType type,
    bool use_int64) {
  std::string kernel_suffix{};
  using namespace std::literals;
  switch (type) {
    case synDataType::syn_type_float: {
      return "f32"sv;
    }
    case synDataType::syn_type_fp16: {
      return "f16"sv;
    }
    case synDataType::syn_type_fp8_152: {
      return "f8"sv;
    }
    case synDataType::syn_type_fp8_143: {
      return "hf8"sv;
    }
    case synDataType::syn_type_int8: {
      return "i8"sv;
    }
    case synDataType::syn_type_uint8: {
      return "u8"sv;
    }
    case synDataType::syn_type_int16: {
      return "i16"sv;
    }
    case synDataType::syn_type_int32: {
      return "i32"sv;
    }
    case synDataType::syn_type_int64: {
      if (use_int64) {
        return "i64"sv;
      } else {
        // Temporary solution: To use autocast feature from complex guid, we
        // need to call _i32 version of the kernel, but pass i64 tensors.
        // Instead of changing guid names in every op implementation, it was
        // changed here.
        return "i32"sv;
      }
    }
    case synDataType::syn_type_bf16: {
      return "bf16"sv;
    }
    default: {
      HABANA_ASSERT(
          false, "Error getting suffix/precision type: Unknown type.");
    }
  }
}

uint64_t graph::recipe_handle::get_recipe_host_mem_size() {
  if (recipe_size_ != 0)
    return recipe_size_;
  synRecipeAttribute recipe_attr(RECIPE_ATTRIBUTE_HOST_MEM_SIZE);
  auto status = synRecipeGetAttribute(
      (&recipe_size_), &recipe_attr, 1, syn_recipe_handle_);
  if (status != synSuccess)
    PT_SYNHELPER_WARN(
        Logger::formatStatusMsg(status), "Failed to retrieve recipe size");
  return recipe_size_;
}

namespace {
void SynapseRecipeDestroyTask(synRecipeHandle recipeHandle) {
  if (recipeHandle && synRecipeDestroy(recipeHandle) != synStatus::synSuccess) {
    PT_SYNHELPER_WARN("Failed to destroy recipe: ", recipeHandle);
  }
}
} // namespace

graph::recipe_handle::~recipe_handle() {
  if (syn_recipe_handle_) {
    habana_helpers::Singleton_GarbageCollectionThreadPool::getInstance()
        .Enqueue(SynapseRecipeDestroyTask, std::move(syn_recipe_handle_));
  }
}

void graph::collect_dst_synapse_nodes(
    graph::Op2NodeContainer::mapped_type& dst_synapse_node_ids,
    const std::string& dst_node) {
  absl::flat_hash_map<std::string, bool> visited_nodes;
  collect_dst_synapse_nodes(dst_synapse_node_ids, dst_node, visited_nodes);
}

void graph::collect_dst_synapse_nodes(
    graph::Op2NodeContainer::mapped_type& dst_synapse_node_ids,
    const std::string& dst_node,
    absl::flat_hash_map<std::string, bool>& visited_nodes) {
  bool was_visited;
  bool is_fully_processed;

  {
    auto it = visited_nodes.find(dst_node);
    if (it == visited_nodes.end()) {
      was_visited = false;
      is_fully_processed = false;
    } else {
      was_visited = true;
      is_fully_processed = it->second;
    }
  }

  if (is_fully_processed) {
    // Node is fully processed by DFS-based traversal, there is no cycle and
    // we have nothing to do here.
    return;
  }

  // Safety measure to prevent infinite loop.
  if (was_visited) {
    // Node was visited and is NOT fully processed. It means that DFS-based
    // traversal found a path from node to the node itself. Cycle!
    PT_SYNHELPER_FATAL("Cycle within Synapse graph detected!");
  }
  visited_nodes.emplace(dst_node, false);

  auto op_to_node_iter = op_to_node_container_.find(dst_node);
  if (op_to_node_iter != op_to_node_container_.end() &&
      !op_to_node_iter->second.empty()) {
    dst_synapse_node_ids.insert(
        begin(op_to_node_iter->second), end(op_to_node_iter->second));
    // Marking node as fully processed
    visited_nodes[dst_node] = true;
    return;
  }

  auto control_edges_iter = control_edges_container_.find(dst_node);
  auto data_edges_iter = data_edges_container_.find(dst_node);
  if (control_edges_iter != end(control_edges_container_)) {
    for (const auto& chained_node : control_edges_iter->second) {
      collect_dst_synapse_nodes(
          dst_synapse_node_ids, chained_node, visited_nodes);
    }
  }

  if (data_edges_iter != end(data_edges_container_)) {
    for (const auto& chained_node : data_edges_iter->second) {
      collect_dst_synapse_nodes(
          dst_synapse_node_ids, chained_node, visited_nodes);
    }
  }

  // Marking node as fully processed
  visited_nodes[dst_node] = true;
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
      // Some ops like NoOp do not have underlying synapse nodes - it is
      // handled in collect_dst_synapse_nodes

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
        graph_handle_,
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
      graph_handle_,
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
