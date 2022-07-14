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
  hpu_ops/generated/hpu_op0.cpp
  hpu_ops/generated/hpu_op1.cpp
  hpu_ops/generated/hpu_op2.cpp
  hpu_ops/generated/hpu_op3.cpp
  hpu_ops/generated/hpu_op4.cpp
  hpu_ops/generated/wrap_kernels_registrations.cpp
)
target_sources(habana_pytorch_plugin PRIVATE ${HPU_OP_SOURCES})
