# ##############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited. This file contains Habana Labs, Ltd. proprietary and
# confidential information and is subject to the confidentiality and license
# agreements under which it was provided.
#
# ##############################################################################

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/graph_visualization_pb2.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph_visualization.proto
    COMMAND protoc -I ${CMAKE_CURRENT_SOURCE_DIR} --python_out=${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/graph_visualization.proto
    )

add_custom_target(
    protobuf ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph_visualization_pb2.py
)