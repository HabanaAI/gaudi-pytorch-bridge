/******************************************************************************
 * Copyright (C) 2020,2021 HabanaLabs, Ltd.
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
#include "habana_helpers/logging.h"
#include "habana_helpers/stat_collection.h"
#include "synapse_helpers/device.h"
#include "synapse_helpers/devmem_logger.h"
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/stream.h"
#include "synapse_helpers/tensor_builder_base.h"
#include "synapse_helpers/util.h"
#include "util/time_measure.h"

namespace synapse_helpers {

namespace {

const std::string graph_prefix = ".graph_dumps/";
std::string get_unique_recipe_name(const std::string& name) {
  static uint64_t suffix = -1;
  if ((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
      !(IS_SYNHELPER_DEBUG_ENABLED)) {
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

synapse_error_v<graph> graph::create(
    device& device,
    std::string name,
    bool dry_run) {
  PT_SYNHELPER_BEGIN;
  graph syn_graph(device, std::move(name));

  PT_SYNHELPER_DEBUG("Graph Create.");
  synStatus status = synSuccess;
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2 &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SYN_API) == true) {
    status =
        synGraphCreateEager(&syn_graph.graph_handle_, syn_graph.device_.type());
  } else {
    status = synGraphCreate(&syn_graph.graph_handle_, syn_graph.device_.type());
  }
  SYNAPSE_SUCCESS_CHECK("Graph creation failed.", status)
  syn_graph.is_valid_ = true;
  syn_graph.dry_run_ = dry_run;
  PT_SYNHELPER_END;
  return {std::move(syn_graph)};
}

synapse_error_v<graph> graph::create_for_refinement(
    device& device,
    std::string name) {
  PT_SYNHELPER_BEGIN;
  graph syn_graph(device, std::move(name));

  PT_SYNHELPER_DEBUG("Graph Create.");
  synStatus status = synSuccess;
  status = synGraphCreate(&syn_graph.graph_handle_, syn_graph.device_.type());
  SYNAPSE_SUCCESS_CHECK("Graph creation failed.", status)
  syn_graph.is_valid_ = true;
  syn_graph.dry_run_ = false;
  PT_SYNHELPER_END;
  return {std::move(syn_graph)};
}

graph::graph(graph&& other) noexcept
    : device_{other.device_},
      name_{other.name_},
      is_valid_{other.is_valid_},
      in_build_phase_(other.in_build_phase_),
      in_execution_phase_(other.in_execution_phase_),
      graph_handle_(other.graph_handle_),
      dry_run_(other.dry_run_) {
  other.is_valid_ = false;
  other.graph_handle_ = {};
}

graph::~graph() {
  if (is_valid_) {
    PT_SYNHELPER_DEBUG("Graph destroy.");
    synGraphDestroy(graph_handle_);

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
    const synapse_error_v<std::string>& node_type_or_err,
    synNodeId* ret_node_id,
    const char** input_layouts,
    const char** output_layouts) {
  if (dry_run_) {
    // Lazy mode shape inference call, early return without execution
    return {};
  }
  SYNAPSE_RETURN_IF_ERROR_V(node_type_or_err);
  const auto& node_type{get_value(node_type_or_err)};
  for (auto& tensor : outputs) {
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
      PT_LAZY_DEBUG(
          "synapse output tensors should not carry permutation. Clearing the permutation from synapse output tensor.");
      synTensorPermutation perm;
      perm.dims = 0;
      auto status = synTensorSetPermutation(tensor, &perm);
      if (status != synStatus::synSuccess) {
        PT_SYNHELPER_WARN("Node " + node_type + "  failed.", " Err: ", status);
      }
    }
  }
  PT_BRIDGE_DEBUG("\nAdding Node to graph with guid = ", node_type.c_str());
  if (!in_build_phase_) {
    return synapse_error{"Graph not in build phase.", synStatus::synFail};
  }
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
  if (status != synStatus::synSuccess) {
    PT_SYNHELPER_WARN("Node " + node_type + " add failed.", " Err: ", status);
  }
  HABANA_ASSERT(status == synStatus::synSuccess)
  graph_is_empty_ = false;
  op_to_node_container_pt_["jit_node"].emplace_back(nodeId);
  if (ret_node_id) {
    *ret_node_id = nodeId;
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
  STAT_START(synapse_compilation);

  status = set_synapse_control_edges();
  SYNAPSE_SUCCESS_CHECK("Setting node dependencies failed.", status);

  TIME_MEASURE_VARS;
  START_TIME_MEASURE;
  auto recipe_handle{absl::make_unique<graph::recipe_handle>()};

  auto name = get_unique_recipe_name(name_);
  status = synGraphCompile(
      &recipe_handle->syn_recipe_handle_, graph_handle_, name.c_str(), nullptr);

  SYNAPSE_SUCCESS_CHECK("Graph compile failed.", status);
  END_TIME_MEASURE("Synapse graph compilation took");
  in_execution_phase_ = true;
  recipe_handle->graph_is_empty_ = graph_is_empty_;
  recipe_handle->in_execution_phase_ = true;
  recipe_handle->recipe_name_ = std::move(name);

  STAT_ADD_ATTRIBUTE(
      globalStatPtsEnum::recipe_compile,
      "Recipe Name",
      recipe_handle->recipe_name_);
  STAT_COLLECT_TIME(synapse_compilation, globalStatPtsEnum::recipe_compile);
  return {std::move(recipe_handle)};
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

synapse_error_v<uint64_t> graph::query_workspace_size(
    const graph::recipe_handle& recipe_handle) {
  uint64_t workspace_size;
  SYNAPSE_SUCCESS_CHECK(
      "Getting workspace size failed",
      synWorkspaceGetSize(&workspace_size, recipe_handle.syn_recipe_handle_));
  return workspace_size;
}

synapse_error_o graph::query_recipe_tensor_info(
    std::shared_ptr<graph::recipe_handle> recipe_handle,
    std::vector<synRetrievedLaunchTensorInfo>& tensor_info_vec) {
  SYNAPSE_SUCCESS_CHECK(
      "Getting launch tensor info failed",
      synTensorRetrieveLaunchInfoById(
          recipe_handle->syn_recipe_handle_,
          tensor_info_vec.size(),
          tensor_info_vec.data()));
  return {};
}

synapse_error_o graph::launch(
    device& device,
    const graph::recipe_handle& recipe_handle,
    uint64_t workspace_size,
    std::vector<synLaunchTensorInfo>&& inputs_and_outputs_info,
    std::unique_ptr<device_ptr_lock>& address_lock,
    std::vector<shared_event>& ext_events) {
  return launch(
      device,
      recipe_handle,
      workspace_size,
      inputs_and_outputs_info,
      address_lock,
      ext_events);
}

synapse_error_o graph::launch(
    device& device,
    const graph::recipe_handle& recipe_handle,
    uint64_t workspace_size,
    std::vector<synLaunchTensorInfo>& inputs_and_outputs_info,
    std::unique_ptr<device_ptr_lock>& address_lock,
    std::vector<shared_event>& ext_events) {
  PT_SYNHELPER_BEGIN;
  synStatus status;

  if (recipe_handle.graph_is_empty_) {
    // Valid case, in some special scenarios Op does not add to graph.
    return {};
  }

  if (!recipe_handle.in_execution_phase_) {
    return synapse_error{"Graph not in execution phase.", synStatus::synFail};
  }
  PT_SYNHELPER_DEBUG(
      "in graph::launch, launch handle string:\n",
      absl::StrFormat(
          "------Launch-handle %s------\n"
          "input_outputs_names={%s}\n"
          "-------------------------",
          recipe_handle.recipe_name_,
          to_string(inputs_and_outputs_info)));

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
  SYNAPSE_RETURN_IF_ERROR(
      std::find_if(
          inputs_and_outputs_info.begin(),
          inputs_and_outputs_info.end(),
          table_checker) == inputs_and_outputs_info.end());
  auto& compute_stream = device.get_compute_stream();

  auto workspace_buffer = device.get_workspace_buffer(workspace_size);
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
  {
    address_lock = absl::make_unique<device_ptr_lock>(
        device.lock_addresses(absl::Span<const device_ptr>(addresses)));
    auto iter = inputs_and_outputs_info.begin();
    size_t index = 0;
    for (auto address : *address_lock) {
      if (host_address_map.count(index)) {
        iter->pTensorAddress = host_address_map[index];
      } else {
        iter->pTensorAddress = address;
      }
      ++index;
      ++iter;
    }

    if (GET_ENV_FLAG_NEW(PT_HABANA_MEM_LOG_LEVEL) == MEM_LOG_GRAPH_LAUNCH) {
      std::string msg = absl::StrFormat(
          "%s%s", "Before launch of graph", recipe_handle.recipe_name_.c_str());
      synapse_helpers::print_live_allocations(msg.c_str());
    }

    uint32_t flags{0};
    std::vector<synEventHandle> event_handles;
    event_handles.reserve(ext_events.size());
    std::transform(
        ext_events.begin(),
        ext_events.end(),
        std::back_inserter(event_handles),
        [](shared_event& event) -> synEventHandle { return *event; });

    status = synLaunchWithExternalEventsBase(
        compute_stream,
        inputs_and_outputs_info.data(),
        inputs_and_outputs_info.size(),
        workspace_buffer,
        recipe_handle.syn_recipe_handle_,
        event_handles.data(),
        event_handles.size(),
        flags);
  }

  SYNAPSE_SUCCESS_CHECK("synLaunch failed.", status)
  PT_SYNHELPER_END;

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

uint64_t graph::recipe_handle::get_recipe_host_mem_size() {
  if (recipe_size_ != 0)
    return recipe_size_;
  synRecipeAttribute recipe_attr(RECIPE_ATTRIBUTE_HOST_MEM_SIZE);
  auto status = synRecipeGetAttribute(
      (&recipe_size_), &recipe_attr, 1, syn_recipe_handle_);
  if (status != synSuccess)
    PT_SYNHELPER_WARN("Failed to retrieve recipe size");
  return recipe_size_;
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
    // Node is fully processed by DFS-based traversal, there is no cycle and we
    // have nothing to do here.
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
