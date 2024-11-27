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

%module py_synapse_logger

%{
#define SWIG_FILE_WITH_INIT
#include "py_synapse_logger.h"
using namespace synapse_logger;
%}

%include "std_string.i"
%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* IN_ARRAY1, int DIM1) {(float* vec, int n)}

%include "py_synapse_logger.h"
