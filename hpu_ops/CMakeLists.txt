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

file(GLOB HPU_OP_SOURCES *.cpp)
list(APPEND HPU_OP_SOURCES
  ${CMAKE_BINARY_DIR}/generated/hpu_op0.cpp
  ${CMAKE_BINARY_DIR}/generated/hpu_op1.cpp
  ${CMAKE_BINARY_DIR}/generated/hpu_op2.cpp
  ${CMAKE_BINARY_DIR}/generated/hpu_op3.cpp
  ${CMAKE_BINARY_DIR}/generated/hpu_op4.cpp
  ${CMAKE_BINARY_DIR}/generated/hpu_autocast_ops.cpp
  ${CMAKE_BINARY_DIR}/generated/wrap_kernels_registrations.cpp
)
target_sources(habana_pytorch_plugin PRIVATE ${HPU_OP_SOURCES})
