/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/jit/ir/ir.h>

namespace habana {
namespace custom_op {

using output_shape = std::vector<int64_t>;
using compute_output_shape_function =
    std::function<output_shape(const at::Stack& stack)>;

using user_param = std::shared_ptr<void>;
using allocate_user_params_func =
    std::function<user_param(const at::Stack& stack, size_t& size)>;

struct NodeDesc {
  // Unique TPC kernel guid
  std::string tpc_guid;
  // jit name as used in TORCH_LIBRARY
  std::string schema_name;
  // TPC kernel params
  allocate_user_params_func user_param_func = nullptr;
};

enum class input_type { TENSOR, SCALAR, USER_PARAMS };

struct InputDesc {
  input_type type;
  unsigned index;
};

struct OutputDesc {
  unsigned index;
  c10::ScalarType dtype = c10::ScalarType::Float;
  compute_output_shape_function compute_output_shape_func = nullptr;
};

class HabanaCustomOpDescriptor {
 public:
  HabanaCustomOpDescriptor(
      NodeDesc node_desc,
      const std::vector<InputDesc>& inputs,
      const std::vector<OutputDesc>& outputs)
      : node_desc_(node_desc), inputs_(inputs), outputs_(outputs) {}
  HabanaCustomOpDescriptor() {}

  std::vector<at::Tensor> execute(const std::vector<c10::IValue>& inputs);

  std::string getSchemaName() const;
  std::string getGuid() const;
  unsigned getInputsSize() const;
  unsigned getOutputsSize() const;
  const std::vector<InputDesc>& getInputs() const;
  const std::vector<OutputDesc>& getOutputs() const;
  bool hasUserParamsFunc() const;
  const allocate_user_params_func& getUserParamsAllocFunc() const;
  bool hasOutputShapeFunc(unsigned index) const;
  const compute_output_shape_function& getOutputShapeFunc(unsigned index) const;

 private:
  NodeDesc node_desc_;
  std::vector<InputDesc> inputs_;
  std::vector<OutputDesc> outputs_;
};

void registerKernel(habana::custom_op::HabanaCustomOpDescriptor& new_desc);

#define REGISTER_CUSTOM_OP_ATTRIBUTES(                                    \
    schema_name, guid, input_desc, output_desc, param_func)               \
  {                                                                       \
    habana::custom_op::NodeDesc node_desc{guid, schema_name, param_func}; \
    habana::custom_op::HabanaCustomOpDescriptor op_desc{                  \
        node_desc, inputs_desc, outputs_desc};                            \
    habana::custom_op::registerKernel(op_desc);                           \
  }

#define HPU_PARAMS_STUB(structname) \
  size = sizeof(structname);        \
  auto params = std::make_shared<structname>()

} // namespace custom_op
} // namespace habana