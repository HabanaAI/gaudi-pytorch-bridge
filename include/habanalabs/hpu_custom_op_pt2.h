/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>

namespace habana {

/**
 * Structure defining output tensor metadata.
 */
struct PartialOutputMetaData {
  at::ScalarType dtype{at::ScalarType::Undefined};
  std::vector<int64_t> shape{};
};

using PartialOutputMetaDataVector = std::vector<PartialOutputMetaData>;

namespace custom_op {

/**
 * Callback used for computing output tensors metadata.
 *
 * @param stack Current operation stack as passed from PyTorch.
 *
 * @return Vector of output tensors metadata.
 */
using OutputMetaFn =
    std::function<PartialOutputMetaDataVector(const at::Stack&)>;

/**
 * Callback used for allocating and filling TPC user params structure.
 *
 * @param stack Current operation stack as passed from PyTorch.
 * @param[out] size For returning allocated TPC params structure size.
 *
 * @return Pointer to TPC user params structure.
 */
using FillParamsFn =
    std::function<std::shared_ptr<void>(const at::Stack&, size_t&)>;

/**
 * Descriptor for custom op containing all necessary information to
 * define user HPU TPC kernel.
 *
 * @param schema Name of registered torch operator
 * @param guid Name of TPC kernel called by the operator
 * @param output_meta_fn Callback for output metadata calculation
 * @param fill_params_fn Callback filling TPC kernel params structure if
 * necessary
 */
class UserCustomOpDescriptor {
 public:
  UserCustomOpDescriptor(
      const std::string& schema,
      const std::string& guid,
      OutputMetaFn output_meta_fn,
      FillParamsFn fill_params_fn = nullptr)
      : schema_(schema),
        guid_(guid),
        output_meta_fn_(output_meta_fn),
        fill_params_fn_(fill_params_fn) {}
  UserCustomOpDescriptor() {}

  /**
   * Actual call by user C++ to op
   *
   * @param inputs All values by order to op execution
   *
   * @return Vector of op results.
   */
  std::vector<at::Tensor> execute(const std::vector<c10::IValue>& inputs);

  /**
   * Get the Custom Op Descriptor object
   *
   * @param op schema registration name which is used in
   * registerUserCustomOp
   *
   * @return Custom op descriptor.
   */
  static const UserCustomOpDescriptor& getUserCustomOpDescriptor(
      const std::string& op);

  /**
   * Schema name as used in TORCH_LIBRARY.
   */
  const std::string& getSchemaName() const;

  /**
   * TPC kernel GUID.
   */
  const std::string& getGuid() const;

  /**
   * Callback to calculate output tensors metadata
   */
  const OutputMetaFn& getOutputMetaFn() const;

  /**
   * Callback to allocate and set user params.
   */
  const FillParamsFn& getFillParamsFn() const;

 private:
  /**
   * Schema name as used in TORCH_LIBRARY.
   */
  std::string schema_;

  /**
   * TPC kernel name.
   */
  std::string guid_;

  /**
   * Callback to calculate output tensors metadata
   */
  OutputMetaFn output_meta_fn_{nullptr};

  /**
   * Callback to allocate and set user params.
   */
  FillParamsFn fill_params_fn_{nullptr};
};

/**
 * Add custom op to kernel registry and expose it to PyTorch.
 *
 * @param schema_name Schema name as set in TORCH_LIBRARY.
 * @param guid TPC kernel guid.
 * @param output_meta_fn Function specifying output tensors.
 * @param fill_params_fn Function filling kernel's params.
 */
void registerUserCustomOp(
    const std::string& schema,
    const std::string& guid,
    OutputMetaFn output_meta_fn,
    FillParamsFn fill_params_fn);

/**
 * Helper macro to shorten allocating TPC user params structure and set size
 * output parameter.
 * To be used inside fill_params_fn callback.
 */
#define HPU_PARAMS_STUB(struct_name) \
  size = sizeof(struct_name);        \
  auto params = std::make_shared<struct_name>()

} // namespace custom_op
} // namespace habana