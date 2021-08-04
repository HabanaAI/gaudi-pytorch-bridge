###############################################################################
# Copyright (C) 2021 HabanaLabs Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################

file(GLOB_RECURSE HPU_OP_SOURCES *.cpp)
list(APPEND HPU_OP_SOURCES hpu_ops/generated/hpu_op.cpp)
target_sources(habana_pytorch_plugin PRIVATE ${HPU_OP_SOURCES})
