/*******************************************************************************
 * From PyTorch:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon
 *Bottou, Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research
 *Institute (Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute
 *(Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 *
 * From Caffe2:
 *
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 *
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 *
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 *
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 *
 * All contributions by Kakao Brain:
 * Copyright 2019-2020 Kakao Brain
 *
 * All contributions by Cruise LLC:
 * Copyright (c) 2022 Cruise LLC.
 * All rights reserved.
 *
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 *
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 *
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to
 *further mark their specific copyright on a particular contribution, they
 *should indicate their copyright solely in the commit message of the change
 *when it is committed.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC
 *Laboratories America and IDIAP Research Institute nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************
 *
 * The following file is a modification of a file found in the PyTorch
 *repository. The modification includes rewriting the partitioner logic from
 *Python to c++
 * -
 *https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/infra/partitioner.py
 */
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

#include <memory>
#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <pybind11/chrono.h>
#include <pybind11/stl.h>
#include <absl/container/flat_hash_set.h>

#define NO_PARTITION -1

namespace py = pybind11;

template <typename T>
class InsertionOrderUnorderedSet {
 private:
  absl::flat_hash_set<T> container;
  std::vector<T> tracker;

 public:

  InsertionOrderUnorderedSet() = default;

  InsertionOrderUnorderedSet(const InsertionOrderUnorderedSet& other)
        : container(other.container), tracker(other.tracker) {}

  InsertionOrderUnorderedSet(InsertionOrderUnorderedSet&& other) noexcept
        : container(std::move(other.container)), tracker(std::move(other.tracker)) {}

  InsertionOrderUnorderedSet& operator=(const InsertionOrderUnorderedSet& other) {
    if (this != &other) {
      container = other.container;
      tracker = other.tracker;
    }
    return *this;
  }

  InsertionOrderUnorderedSet& operator=(InsertionOrderUnorderedSet&& other) noexcept {
    if (this != &other) {
      container = std::move(other.container);
      tracker = std::move(other.tracker);
    }
    return *this;
  }

  void reserve(size_t n) {
    tracker.reserve(n);
    container.reserve(n);
  }

  void insert(const T& el) {
    if (container.insert(el).second)
      tracker.push_back(el);
  }

  template <typename InputIt>
  void insert(InputIt first, InputIt last) {
    for (auto it = first; it != last; ++it) {
      insert(*it);
    }
  }

  void erase(const T& el) {
    auto it = container.find(el);
    if (it != container.end()) {
      container.erase(it);
      const auto& tracker_it = std::find(tracker.begin(), tracker.end(), el);
      if (tracker_it != tracker.end())
        tracker.erase(tracker_it);
    }
  }

  bool exist(const T& el) const {
    return container.find(el) != container.end();
  }

  void clear() {
        container.clear();
        tracker.clear();
    }

  size_t size() const {
    return container.size();
  }

  typename std::vector<T>::iterator begin() {
    return tracker.begin();
  }

  typename std::vector<T>::iterator end() {
    return tracker.end();
  }

  typename std::vector<T>::const_iterator begin() const {
    return tracker.begin();
  }

  typename std::vector<T>::const_iterator end() const {
    return tracker.end();
  }
};

template <typename TKey, typename TValue>
class InsertionOrderUnorderedMap {
 private:
  std::unordered_map<TKey, TValue> container;
  std::vector<TKey> tracker;

 public:
  void insert(const TKey& key, const TValue& value) {
    if (!exist(key)) {
      container.insert({key, value});
      tracker.push_back(key);
    } else
      container[key] = value;
  }

  void erase(const TKey& key) {
    auto it = container.find(key);
    if (it != container.end()) {
      container.erase(it);
      const auto& tracker_it = std::find(tracker.begin(), tracker.end(), key);
      if (tracker_it != tracker.end())
        tracker.erase(tracker_it);
    }
  }

  bool exist(const TKey& key) const {
    return container.find(key) != container.end();
  }

  typename std::unordered_map<TKey, TValue>::iterator get(const TKey& key) {
    return container.find(key);
  }

  size_t size() const {
    return container.size();
  }

  typename std::vector<TKey>::iterator begin() {
    return tracker.begin();
  }

  typename std::vector<TKey>::iterator end() {
    return tracker.end();
  }

  typename std::vector<TKey>::const_iterator begin() const {
    return tracker.begin();
  }

  typename std::vector<TKey>::const_iterator end() const {
    return tracker.end();
  }
};

class Node {
 private:
  int _prim_id;
  std::string _name;
  std::string _op;
  std::string _target_qualified_name;
  bool _is_target_callable; // callable(node.target)
  bool _is_supported;
  InsertionOrderUnorderedSet<Node*> _users;
  InsertionOrderUnorderedSet<Node*> _input_nodes;

 public:
  Node(
      int _prim_id,
      std::string _name,
      std::string _op,
      std::string _target_qualified_name,
      bool _is_target_callable,
      bool _is_supported)
      : _prim_id(_prim_id),
        _name(_name),
        _op(_op),
        _target_qualified_name(_target_qualified_name),
        _is_target_callable(_is_target_callable),
        _is_supported(_is_supported) {
    std::vector<std::string> acceptable_ops = {
        "placeholder",
        "call_method",
        "call_module",
        "call_function",
        "get_attr",
        "output",
        "root"};
    bool is_acceptable =
        (std::find(acceptable_ops.begin(), acceptable_ops.end(), _op) !=
         acceptable_ops.end());
    if (!is_acceptable)
      throw std::invalid_argument("received not acceptable op name");
  }

  int prim_id() const {
    return _prim_id;
  }
  const std::string& name() const {
    return _name;
  }
  const std::string& target_qualified_name() const {
    return _target_qualified_name;
  }
  const std::string& op() const {
    return _op;
  }
  bool is_target_callable() const {
    return _is_target_callable;
  }
  bool is_supported() const {
    return _is_supported;
  }
  InsertionOrderUnorderedSet<Node*>& users() {
    return _users;
  }
  InsertionOrderUnorderedSet<Node*>& all_input_nodes() {
    return _input_nodes;
  }
};

struct PartitionDTO {
  int id;
  std::vector<int> nodes_ids;
};

class Partition {
 private:
  int _id;
  InsertionOrderUnorderedSet<Node*> _nodes;

 public:
  template <typename Iterable>
  Partition(int _id, const Iterable& nodes_iterable) : _id(_id) {
    static_assert(
        std::is_same<
            typename std::iterator_traits<
                typename Iterable::iterator>::value_type, Node*>::value,
        "Iterable must be a container of Node* type");
    for (auto& item : nodes_iterable) {
      _nodes.insert(item);
    }
  }

  void add_node(Node* node) {
    _nodes.insert(node);
  }

  void remove_node(Node* node) {
    _nodes.erase(node);
  }

  int size() const {
    return _nodes.size();
  }

  PartitionDTO convert_to_dto() const {
    std::vector<int> nodes_ids;
    for (auto& node_ptr : _nodes)
      nodes_ids.push_back(node_ptr->prim_id());
    return PartitionDTO{_id, nodes_ids};
  }

  const int& id() const {
    return _id;
  }

  InsertionOrderUnorderedSet<Node*>& nodes() {
    return _nodes;
  }
};

class DependencyViewer {
 private:
  std::unordered_map<Node*, absl::flat_hash_set<Node*>> _upstreams;
  std::unordered_map<Node*, absl::flat_hash_set<Node*>> _downstreams;

 public:
  DependencyViewer(std::vector<Node>& nodes) {
    for (Node& node_ref : nodes) {
      Node* node = &node_ref;
      for (Node* input_node : node->all_input_nodes()) {
        _upstreams[node].insert(input_node);
        _upstreams[node].insert(
            _upstreams[input_node].begin(), _upstreams[input_node].end());
      }
    }

    for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
      Node* node = &(*it);
      for (Node* output_node : node->users()) {
        _downstreams[node].insert(output_node);
        _downstreams[node].insert(
            _downstreams[output_node].begin(), _downstreams[output_node].end());
      }
    }
  }

  absl::flat_hash_set<Node*>& downstreams_of(Node* node) {
    return _downstreams[node];
  }

  absl::flat_hash_set<Node*>& upstreams_of(Node* node) {
    return _upstreams[node];
  }
};

class BindedPartitioner {
 private:
  std::vector<Node> _nodes;
  bool _allows_single_node_partition;
  std::vector<std::string> _non_compute_ops;
  std::vector<std::string> _allowed_single_node_partition_ops;
  std::unique_ptr<DependencyViewer> _dependency_viewer;

 public:
  BindedPartitioner(
      py::list& node_wrappers,
      bool _allows_single_node_partition,
      std::vector<std::string> _non_compute_ops,
      std::vector<std::string> _allowed_single_node_partition_ops)
      : _allows_single_node_partition(_allows_single_node_partition),
        _non_compute_ops(_non_compute_ops),
        _allowed_single_node_partition_ops(_allowed_single_node_partition_ops) {
    std::unordered_map<int, Node*> mapping;
    _nodes.reserve(node_wrappers.size());

    for (auto& node_wrapper : node_wrappers) {
      int prim_id = node_wrapper.attr("prim_id").cast<int>();

      _nodes.emplace_back(
          prim_id,
          node_wrapper.attr("name").cast<std::string>(),
          node_wrapper.attr("op").cast<std::string>(),
          node_wrapper.attr("target_qualified_name").cast<std::string>(),
          node_wrapper.attr("is_target_callable").cast<bool>(),
          node_wrapper.attr("is_supported").cast<bool>());
      mapping.insert({prim_id, &_nodes.back()});
    }

    for (auto& node_wrapper : node_wrappers) {
      int prim_id = node_wrapper.attr("prim_id").cast<int>();
      Node* ptr = mapping.find(prim_id)->second;

      std::vector<int> users =
          node_wrapper.attr("users").cast<std::vector<int>>();
      for (int& id : users)
        ptr->users().insert(mapping.find(id)->second);

      std::vector<int> input_nodes =
          node_wrapper.attr("input_nodes").cast<std::vector<int>>();
      for (int& id : input_nodes)
        ptr->all_input_nodes().insert(mapping.find(id)->second);
    }

    _dependency_viewer = std::make_unique<DependencyViewer>(_nodes);
  }

  std::vector<PartitionDTO> propose_partitions() {
    std::unordered_map<int, absl::flat_hash_set<int>> partition_map;
    InsertionOrderUnorderedMap<Node*, int> assignment;
    InsertionOrderUnorderedMap<int, std::shared_ptr<Partition>> partitions_by_id;
    int new_partition_id = 0;

    for (auto it = _nodes.rbegin(); it != _nodes.rend(); it++) {
      Node* node = &(*it);
      InsertionOrderUnorderedSet<int> merge_candidates;

      if (node->is_supported() && assignment.exist(node) == false) {
        int partition_id = new_partition_id++;
        merge_single_node(
            node, partition_id, partition_map, assignment, partitions_by_id);
        merge_candidates.insert(partition_id);
      }

      for (auto it = assignment.begin(); it != assignment.end(); it++)
        merge_candidates.insert(assignment.get(*it)->second);

      if (merge_candidates.size() > 1) {
        auto it = merge_candidates.begin();
        const int& self_id = *it++;
        for (; it != merge_candidates.end(); it++)
          maybe_merge_partition(
              self_id, *it, partition_map, assignment, partitions_by_id);
      }
    }

    std::unordered_map<Node*, int> nodes_reassignment;
    for (Node& node_ref : _nodes) {
      Node* node = &node_ref;
      bool is_tuple_output = true;
      for (Node* user : node->users()) {
        if (user->op() != "call_function" ||
            user->target_qualified_name() != "_operator.getitem") {
          is_tuple_output = false;
          break;
        }
      }

      if (is_tuple_output) {
        int id = assignment.exist(node) ? assignment.get(node)->second
                                        : NO_PARTITION;
        for (Node* user : node->users()) {
          int assignment_id = assignment.exist(user)
              ? assignment.get(user)->second
              : NO_PARTITION;
          if (id != assignment_id)
            nodes_reassignment[user] = id;
        }
      }
    }

    for (auto it = nodes_reassignment.begin(); it != nodes_reassignment.end();
         it++)
      merge_single_node(
          it->first, it->second, partition_map, assignment, partitions_by_id);

    if (_allows_single_node_partition == false) {
      std::vector<std::string> default_non_compute_ops = {
          "torch.ops.aten.view", "_operator.getitem"};
      std::vector<int> partitions_to_remove;
      for (auto& it : partitions_by_id) {
        int id = it;
        std::shared_ptr<Partition> partition = partitions_by_id.get(id)->second;
        int compute_node_count = 0;
        for (Node* node : partition->nodes()) {
          if (node->op() == "call_function") {
            assert(node->is_target_callable());
            if (std::find(
                    _non_compute_ops.begin(),
                    _non_compute_ops.end(),
                    node->target_qualified_name()) == _non_compute_ops.end() &&
                std::find(
                    default_non_compute_ops.begin(),
                    default_non_compute_ops.end(),
                    node->target_qualified_name()) ==
                    default_non_compute_ops.end()) {
              compute_node_count += 1;
            }
            if (std::find(
                    _allowed_single_node_partition_ops.begin(),
                    _allowed_single_node_partition_ops.end(),
                    node->target_qualified_name()) !=
                _allowed_single_node_partition_ops.end())
              compute_node_count += 1;
          }
        }
        if (compute_node_count <= 1)
          partitions_to_remove.push_back(id);
      }
      for (int& id : partitions_to_remove)
        partitions_by_id.erase(id);
    }

    std::vector<PartitionDTO> proposed_partitions;
    for (auto& it : partitions_by_id)
      proposed_partitions.push_back(partitions_by_id.get(it)->second->convert_to_dto());

    return proposed_partitions;
  }

 private:
  bool dfs_iter_find_cycle(
      const int self_id,
      const int other_id,
      absl::flat_hash_set<Node*>& all_user_nodes,
      InsertionOrderUnorderedSet<Node*>& merged_nodes,
      std::unordered_map<int, absl::flat_hash_set<int>>& partition_map,
      InsertionOrderUnorderedMap<Node*, int>& assignment) {
    for (Node* user_node : all_user_nodes) {
      absl::flat_hash_set<int> visited_partition_ids;

      for (Node* path_node : _dependency_viewer->downstreams_of(user_node)) {
        if (merged_nodes.exist(path_node))
          return true;

        if (assignment.exist(path_node)) {
          int partition_id = assignment.get(path_node)->second;

          if (visited_partition_ids.find(partition_id) !=
              visited_partition_ids.end())
            continue;

          absl::flat_hash_set<int>& p_map = partition_map[partition_id];
          if (p_map.find(self_id) != p_map.end() ||
              p_map.find(other_id) != p_map.end())
            return true;
          visited_partition_ids.insert(partition_id);
        }
      }
    }
    return false;
  }

  bool maybe_merge_partition(
      const int self_id,
      const int other_id,
      std::unordered_map<int, absl::flat_hash_set<int>>& partition_map,
      InsertionOrderUnorderedMap<Node*, int>& assignment,
      InsertionOrderUnorderedMap<int, std::shared_ptr<Partition>>& partitions_by_id) {
    InsertionOrderUnorderedSet<Node*>& self_nodes = partitions_by_id.get(self_id)->second->nodes();
    InsertionOrderUnorderedSet<Node*>& other_nodes =
        partitions_by_id.get(other_id)->second->nodes();
    InsertionOrderUnorderedSet<Node*> merged_nodes;
    merged_nodes.reserve(self_nodes.size() + other_nodes.size());
    merged_nodes.insert(self_nodes.begin(), self_nodes.end());
    merged_nodes.insert(other_nodes.begin(), other_nodes.end());

    absl::flat_hash_set<Node*> all_user_nodes;
    for (Node* node : merged_nodes)
      for (Node* user_node : node->users())
        if (!merged_nodes.exist(user_node))
          all_user_nodes.insert(user_node);

    if (dfs_iter_find_cycle(
            self_id,
            other_id,
            all_user_nodes,
            merged_nodes,
            partition_map,
            assignment))
      return false;

    partitions_by_id.get(self_id)->second->nodes() = std::move(merged_nodes);
    merged_nodes.clear();
    for (Node* node : partitions_by_id.get(other_id)->second->nodes())
      assignment.insert(node, self_id);

    partitions_by_id.erase(other_id);

    partition_map[self_id].insert(
        partition_map[other_id].begin(), partition_map[other_id].end());
    partition_map.erase(other_id);

    return true;
  };

  void update_partition_map(
      Node* node,
      int id,
      std::unordered_map<int, absl::flat_hash_set<int>>& partition_map,
      InsertionOrderUnorderedMap<Node*, int>& assignment) {
    const absl::flat_hash_set<Node*>& downstream_nodes =
        _dependency_viewer->downstreams_of(node);
    for (Node* curr_node : downstream_nodes) {
      if (assignment.exist(curr_node))
        partition_map[id].insert(assignment.get(curr_node)->second);
    }

    const absl::flat_hash_set<Node*>& upstreams_nodes =
        _dependency_viewer->upstreams_of(node);
    for (Node* curr_node : upstreams_nodes) {
      if (assignment.exist(curr_node))
        partition_map[assignment.get(curr_node)->second].insert(id);
    }
  }

  void merge_single_node(
      Node* node,
      int id,
      std::unordered_map<int, absl::flat_hash_set<int>>& partition_map,
      InsertionOrderUnorderedMap<Node*, int>& assignment,
      InsertionOrderUnorderedMap<int, std::shared_ptr<Partition>>& partitions_by_id) {
    if (assignment.exist(node))
      partitions_by_id.get(assignment.get(node)->second)->second->remove_node(node);

    if (id == NO_PARTITION)
      assignment.erase(node);
    else {
      assignment.insert(node, id);
      if (partitions_by_id.exist(id)) {
        partitions_by_id.get(id)->second->add_node(node);
        update_partition_map(node, id, partition_map, assignment);
      } else {
        partitions_by_id.insert(id, std::make_shared<Partition>(id, std::vector<Node*>{node}));
        update_partition_map(node, id, partition_map, assignment);
      }
    }
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PartitionDTO>(m, "PartitionDTO")
      .def_readwrite("id", &PartitionDTO::id)
      .def_readwrite("nodes_ids", &PartitionDTO::nodes_ids);

  py::class_<BindedPartitioner>(m, "BindedPartitioner")
      .def(py::init<
           py::list&,
           bool,
           std::vector<std::string>,
           std::vector<std::string>>())
      .def("propose_partitions", &BindedPartitioner::propose_partitions);
}
