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

macro(set_fabi_version fabi_flag)
  execute_process(
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/get_fabi_flag.py ${CMAKE_CXX_COMPILER}
    OUTPUT_VARIABLE out_get_fabi_flag
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(fabi_flag ${out_get_fabi_flag})
endmacro(set_fabi_version)
