###############################################################################
#
#  Copyright (c) 2025 Intel Corporation
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

macro(detect_pt_version)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__.replace('+hpu.git', '.'))"
    OUTPUT_VARIABLE TORCH_VERSION_AND_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REPLACE "." ";" TORCH_VERSION_LIST ${TORCH_VERSION_AND_HASH})
  list(GET TORCH_VERSION_LIST 0 TORCH_VERSION_MAJOR)
  list(GET TORCH_VERSION_LIST 1 TORCH_VERSION_MINOR)
  list(GET TORCH_VERSION_LIST 2 TORCH_VERSION_PATCH)
  list(GET TORCH_VERSION_LIST 3 TORCH_COMMIT_HASH)

  message(
    STATUS
      "PyTorch version detected: ${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}.${TORCH_VERSION_PATCH}+hpu.git${TORCH_COMMIT_HASH}"
  )
endmacro(detect_pt_version)
