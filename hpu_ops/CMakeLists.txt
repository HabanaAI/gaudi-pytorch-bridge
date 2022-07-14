###############################################################################
# Copyright (C) 2021 HabanaLabs Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
###############################################################################

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE HPU_OP_SOURCES *.cpp)
target_sources(habana_pytorch_plugin PRIVATE ${HPU_OP_SOURCES})

execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/gen_op.py
                     --output_dir=${CMAKE_SOURCE_DIR}/hpu_ops/generated
                     --yaml=${CMAKE_SOURCE_DIR}/scripts/hpu_op.yaml
                     ${CMAKE_SOURCE_DIR}/habana_kernels/wrap_kernels_declarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RegistrationDeclarations.h
                     ${TORCH_INSTALL_PREFIX}/include/ATen/RedispatchFunctions.h
                RESULT_VARIABLE _RESULT
)
if(_RESULT)
  message(FATAL_ERROR "gen_op.py failed with: ${_RESULT}")
endif()
