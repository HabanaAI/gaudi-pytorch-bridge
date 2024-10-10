/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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
namespace custom_op {

/**
 * Output tensor shape.
 */
using output_shape = std::vector<int64_t>;

/**
 * Callback used for computing output shape.
 *
 * @param stack Current operation stack as passed from PyTorch.
 *
 * @return Computed output tensor shape.
 */
using compute_output_shape_function =
    std::function<output_shape(const at::Stack& stack)>;

/**
 * Pointer to TPC user params structure.
 */
using user_param = std::shared_ptr<void>;

/**
 * Callback used for allocating and filling TPC user params structure.
 *
 * @param stack Current operation stack as passed from PyTorch.
 * @param[out] size For returning allocated TPC params structure size.
 *
 * @return Pointer to TPC user params structure.
 */
using allocate_user_params_func =
    std::function<user_param(const at::Stack& stack, size_t& size)>;

/**
 * TPC kernel details.
 */
struct NodeDesc {
  /**
   * Unique TPC kernel guid.
   */
  std::string tpc_guid;

  /**
   * Schema name as used in TORCH_LIBRARY.
   */
  std::string schema_name;

  /**
   * TPC kernels params allocation and setting callback function.
   */
  allocate_user_params_func user_param_func = nullptr;
};

/**
 * Describes flavor of input tensor.
 */
enum class input_type {
  /**
   * Device memory allocated tensor
   */
  TENSOR,
  /**
   * Device memory allocated scalar
   */
  SCALAR,
  /**
   * Host memory allocated data, not passed to TPC as input (e.g. can be stored
   * in TPC params structure or used to alter flow)
   */
  USER_PARAMS
};

/**
 * Op input details.
 */
struct InputDesc {
  /**
   * Flavor of the input.
   */
  input_type type;

  /**
   * Input index, shall be unique for each input.
   */
  unsigned index;
};

/**
 * Op output details
 */
struct OutputDesc {
  /**
   * Output index, shall be unique for each input.
   */
  unsigned index;

  /**
   * Output tensor data type.
   * Default is float.
   */
  c10::ScalarType dtype = c10::ScalarType::Float;

  /**
   * Output shape calculation callback function.
   * If none if provided, it is assumed output has the same shape as input.
   */
  compute_output_shape_function compute_output_shape_func = nullptr;
};

/**
 * Descriptor for custom op containing all necessary information to
 * define user HPU TPC kernel.
 *
 * User is responsible to define all node TPC info within NodeDesc, all
 * inputs/outputs info within vectors of InputDesc/OutputDesc. User needs
 * to register descriptor with the macro REGISTER_CUSTOM_OP_ATTRIBUTES. User
 * will retrieve his descriptor from registry, and call execute with inputs.
 */
class HabanaCustomOpDescriptor {
 public:
  HabanaCustomOpDescriptor(
      NodeDesc node_desc,
      const std::vector<InputDesc>& inputs,
      const std::vector<OutputDesc>& outputs)
      : node_desc_(node_desc), inputs_(inputs), outputs_(outputs) {}
  HabanaCustomOpDescriptor() {}

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
   * @param Op schema registration name which is used in
   * REGISTER_CUSTOM_OP_ATTRIBUTES
   *
   * @return Custom op descriptor.
   */
  static const HabanaCustomOpDescriptor getCustomOpDescriptor(std::string op);

  /**
   * Schema name as used in TORCH_LIBRARY.
   */
  const std::string& getSchemaName() const;

  /**
   * TPC kernel GUID.
   */
  const std::string& getGuid() const;

  /**
   * Number of op inputs.
   */
  unsigned getInputsSize() const;

  /**
   * Number of op outputs.
   */
  unsigned getOutputsSize() const;

  /**
   * List of op input descriptors.
   */
  const std::vector<InputDesc>& getInputs() const;

  /**
   * List of op output descriptors.
   */
  const std::vector<OutputDesc>& getOutputs() const;

  /**
   * Whether callback to allocate and set user params was provided.
   */
  bool hasUserParamsFunc() const;

  /**
   * Callback to allocate and set user params.
   */
  const allocate_user_params_func& getUserParamsAllocFunc() const;

  /**
   * Whether callback to compute output shape was provided.
   */
  bool hasOutputShapeFunc(unsigned index) const;

  /**
   * Callback to compute output shape.
   */
  const compute_output_shape_function& getOutputShapeFunc(unsigned index) const;

 private:
  /**
   * TPC kernel details.
   */
  NodeDesc node_desc_;

  /**
   * List of input descriptors.
   */
  std::vector<InputDesc> inputs_;

  /**
   * List of output descriptors.
   */
  std::vector<OutputDesc> outputs_;

  /**
   * Verifies inputs and outputs have unique indexes.
   */
  void verifyInputOutputIndexes();
};

/**
 * Add custom op to kernel registry and expose it to PyTorch.
 */
void registerKernel(habana::custom_op::HabanaCustomOpDescriptor& new_desc);

/**
 * Main macro for user to register custom op for HPU.
 *
 * @param schema_name Schema name as set in TORCH_LIBRARY.
 * @param guid TPC kernel guid.
 * @param inputs_desc List of input descriptors.
 * @param outputs_desc List of output descriptors.
 */
#define REGISTER_CUSTOM_OP_ATTRIBUTES(                                    \
    schema_name, guid, inputs_desc, outputs_desc, param_func)             \
  {                                                                       \
    habana::custom_op::NodeDesc node_desc{guid, schema_name, param_func}; \
    habana::custom_op::HabanaCustomOpDescriptor op_desc{                  \
        node_desc, inputs_desc, outputs_desc};                            \
    habana::custom_op::registerKernel(op_desc);                           \
  }

/**
 * Helper macro to shorten allocating TPC user params structure and set size
 * output parameter.
 * To be used inside allocate_user_params_func callback.
 */
#define HPU_PARAMS_STUB(struct_name) \
  size = sizeof(struct_name);        \
  auto params = std::make_shared<struct_name>()

} // namespace custom_op
} // namespace habana