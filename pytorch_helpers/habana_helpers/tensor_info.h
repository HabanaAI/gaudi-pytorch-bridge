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

#include <iostream>
#include <string>

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>

#include "habana_helpers/logging.h"
#include "synapse_helpers/graph.h"

using IVal = torch::jit::IValue;
using IValPtrShared = std::shared_ptr<IVal>;
using ValPtr = torch::jit::Value*;

void PrintATenTensor(const at::Tensor& a);
void PrintATenTensor(const IValPtrShared& a);

class PtTensorInfo {
 public:
  PtTensorInfo(const IValPtrShared& ivpsh);
  PtTensorInfo(
      const IValPtrShared& ivp,
      const std::string& sn,
      const ValPtr& vp,
      const bool wflag);
  PtTensorInfo(
      const at::Tensor& pt_tensor,
      const std::string& sn,
      const std::string& irn,
      const bool wflag);

  // access functions for read write data members
  void* get_buffer() const {
    return buffer_;
  }
  void set_buffer(void* bp) {
    buffer_ = bp;
  }

  bool is_duplicate() const {
    return is_duplicate_;
  }
  void set_duplicate_flag(bool b) {
    is_duplicate_ = b;
  }

  size_t get_parent_index() const {
    return parent_index_;
  }
  void set_parent_index(size_t i) {
    parent_index_ = i;
  }

  // access functions for read only data members
  bool is_tensor() const {
    return is_tensor_;
  }
  const IVal& get_ivalue() const {
    return iv_;
  }
  const std::string& get_ir_name() const {
    return ir_name_;
  }
  const std::string& get_syn_name() const {
    return syn_name_;
  }
  const char* get_syn_namec_str() const {
    return syn_name_.c_str();
  }
  const std::string& get_shape_str() const {
    return shape_str_;
  }
  unsigned get_numel() const {
    return numel_;
  }
  unsigned get_size() const {
    return size_;
  }
  bool watch_enabled() const {
    return watch_;
  }

  friend std::ostream& operator<<(std::ostream& O, const PtTensorInfo& t);
  const std::vector<int64_t>& get_shape() const {
    return shape_;
  };
  const c10::TensorOptions& get_topts() const {
    return topts_;
  }
  const c10::MemoryFormat& get_mf() {
    return mf_;
  }

  static bool watch_tensor_flag;

 private:
  bool is_tensor_{true};
  IVal iv_{};

  void* buffer_{nullptr};
  std::string ir_name_;
  std::string syn_name_;
  std::string shape_str_;

  unsigned numel_{0};
  unsigned size_{0};

  // Will hold the index of parent tensor info for aliases
  bool is_duplicate_{false};
  size_t parent_index_{ULONG_MAX};
  bool watch_ = false;

  std::vector<int64_t> shape_;
  c10::TensorOptions topts_;
  c10::MemoryFormat mf_;

  void populate_tinfo(
      const at::Tensor& pt_tensor,
      const std::string& irn,
      const std::string& sn,
      const bool wflag);
};
