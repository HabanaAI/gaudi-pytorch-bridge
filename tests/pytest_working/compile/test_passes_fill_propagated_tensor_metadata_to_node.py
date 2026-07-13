###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
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

import torch
from habana_frameworks.torch.dynamo.compile_backend._helpers.helpers import (
    fill_propagated_tensor_metadata_to_node,
    is_opaque_node,
)
from test_utils import compile_function_if_compile_mode
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import register_opaque_type
from torch._opaque_base import OpaqueBase


class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        flag = x == y
        return flag


class _TestOpaqueObj(OpaqueBase):
    """Minimal class registered as opaque type for testing purposes."""

    def __eq__(self, other):
        return isinstance(other, _TestOpaqueObj)

    def __hash__(self):
        return id(self)

    def __fx_repr__(self):
        return ("test_passes_fill_propagated_tensor_metadata_to_node._TestOpaqueObj()", {_TestOpaqueObj})


register_opaque_type(_TestOpaqueObj, typ="value")


def test_fill_propagated_tensor_metadata_to_node():
    model = MyModule().to("hpu")
    compiled_model = compile_function_if_compile_mode(model, dynamic=True)
    # forced dynamic compilation forces occurence of SymBool in this mini example as internal output type
    retval = compiled_model(2, 3)
    assert retval is False


def test_fill_propagated_tensor_metadata_to_node_opaque_object():
    """Opaque objects (e.g. DeviceMesh wrapped in FakeScriptObject) must not
    raise and must set scalar-like metadata + the is_opaque flag."""
    # Build a minimal FX graph with a single placeholder node.
    graph = torch.fx.Graph()
    node = graph.placeholder("opaque_input")
    graph.output(node)

    # Simulate a FakeScriptObject result (wraps an arbitrary object).
    fake_obj = FakeScriptObject(
        wrapped_obj=object(),
        script_class_name="__torch__.test.Opaque",
        x=None,
    )

    fill_propagated_tensor_metadata_to_node(fake_obj, node)

    assert node.meta["output_device"] == torch.device("cpu")
    assert node.meta["output_dtypes"] == [None]
    assert node.meta["output_layouts"] == [None]
    assert node.meta["output_shapes"] == [()]
    assert node.meta["output_strides"] == [()]
    assert node.meta["output_contiguous"] == [None]
    assert node.meta["output_offset"] == [()]
    assert is_opaque_node(node)


def test_fill_propagated_tensor_metadata_to_node_registered_opaque_type():
    """A value whose type is registered via register_opaque_type (but is NOT
    wrapped in FakeScriptObject) must also be handled as opaque."""
    graph = torch.fx.Graph()
    node = graph.placeholder("opaque_registered")
    graph.output(node)

    fill_propagated_tensor_metadata_to_node(_TestOpaqueObj(), node)

    assert node.meta["output_device"] == torch.device("cpu")
    assert node.meta["output_dtypes"] == [None]
    assert node.meta["output_layouts"] == [None]
    assert node.meta["output_shapes"] == [()]
    assert node.meta["output_strides"] == [()]
    assert node.meta["output_contiguous"] == [None]
    assert node.meta["output_offset"] == [()]
    assert is_opaque_node(node)
