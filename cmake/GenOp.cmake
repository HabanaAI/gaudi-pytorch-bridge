###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

function(generate_files output_files flags)
  add_custom_command(
    OUTPUT ${output_files}
    COMMAND
      ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/run_gen_op.py --output_dir=${CMAKE_BINARY_DIR}/generated
      --yaml=${CMAKE_SOURCE_DIR}/scripts/hpu_op.yaml ${TORCH_INSTALL_PREFIX}/include/ATen/RegistrationDeclarations.h
      ${TORCH_INSTALL_PREFIX}/../torchgen/packaged/ATen/native/native_functions.yaml ${flags}
    MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/scripts/run_gen_op.py
    DEPENDS ${CMAKE_SOURCE_DIR}/scripts/gen_op/code_generation.py
            ${CMAKE_SOURCE_DIR}/scripts/gen_op/code_templates.py
            ${CMAKE_SOURCE_DIR}/scripts/gen_op/constants.py
            ${CMAKE_SOURCE_DIR}/scripts/gen_op/custom_ops.py
            ${CMAKE_SOURCE_DIR}/scripts/gen_op/op_validator.py
            ${CMAKE_SOURCE_DIR}/scripts/gen_op/op.py
            ${CMAKE_SOURCE_DIR}/scripts/gen_op/parser.py
            ${CMAKE_SOURCE_DIR}/scripts/gen_op/version_checker.py
            ${CMAKE_SOURCE_DIR}/scripts/gen_op/yaml_context.py
            ${CMAKE_SOURCE_DIR}/scripts/hpu_op.yaml
            ${CMAKE_SOURCE_DIR}/scripts/hpu_op_templates.yaml
            ${TORCH_INSTALL_PREFIX}/include/ATen/RegistrationDeclarations.h
            ${TORCH_INSTALL_PREFIX}/../torchgen/packaged/ATen/native/native_functions.yaml)
endfunction()
