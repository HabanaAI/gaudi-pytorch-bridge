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

from test_infra_hpu import pytorch_version

pt_dir = pytorch_version

# The file contains gaudi only failures
dynamo_tests_to_deselect = {}

distributed_tests_to_deselect = {}

transformers_tests_to_deselect = {}

autograd_tests_to_deselect = {}

binary_ufuncs_tests_to_deselect = {}

complex_tests_to_deselect = {}

foreach_tests_to_deselect = {}
fx_tests_to_deselect = {}
indexing_tests_to_deselect = {}

linalg_tests_to_deselect = {}

masked_tests_to_deselect = {}

module_init_tests_to_deselect = {}

modules_tests_to_deselect = {}
native_mha_tests_to_deselect = {}
nestedtensor_tests_to_deselect = {}
nn_tests_to_deselect = {}

numpy_interop_tests_to_deselect = {}

ops_tests_to_deselect = {}
ops_jit_tests_to_deselect = {}
op_aliases_tests_to_deselect = {}

reductions_tests_to_deselect = {}

scatter_gather_ops_tests_to_deselect = {}

segment_reductions_tests_to_deselect = {}

serialization_tests_to_deselct = {}

shape_ops_tests_to_deselect = {}

sort_and_select_tests_to_deselect = {}

sparse_tests_to_deselect = {}

sparse_csr_tests_to_deselect = {}

spectral_ops_tests_to_deselect = {}

tensor_creation_ops_tests_to_deselect = {}

testing_tests_to_deselect = {}

torch_tests_to_deselect = {}

type_promotion_tests_to_deselect = {}

unary_ufuncs_tests_to_deselect = {}

view_ops_tests_to_deselect = {}

vmap_tests_to_deselect = {}
expanded_weights_tests_to_deselect = {}
ops_gradients_tests_to_deselect = {}
ops_fwd_gradients_tests_to_deselect = {}
decomp_tests_to_deselect = {}
