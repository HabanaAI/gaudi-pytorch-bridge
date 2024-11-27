###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/graph_visualization_pb2.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph_visualization.proto
    COMMAND protoc -I ${CMAKE_CURRENT_SOURCE_DIR} --python_out=${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/graph_visualization.proto
    )

add_custom_target(
    protobuf ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/graph_visualization_pb2.py
)