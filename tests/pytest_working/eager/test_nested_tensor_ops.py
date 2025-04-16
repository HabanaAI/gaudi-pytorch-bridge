###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

import pytest
import torch


@pytest.fixture(scope="module")
def nested_tensor():
    device = torch.device("hpu")
    return torch.nested.nested_tensor([torch.arange(12).reshape(2, 6)], dtype=torch.float, device=device)


def test_masked_fill_scalar(nested_tensor):
    mask = torch.nested.nested_tensor(
        [torch.tensor([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=torch.bool, device="hpu")]
    )
    result = nested_tensor.masked_fill(mask, -1.0)
    expected = nested_tensor[0].clone()
    expected[mask[0]] = -1.0
    assert torch.allclose(result[0], expected)


def test_eq_scalar(nested_tensor):
    scalar = 5.0
    result = nested_tensor.eq(scalar)
    expected = nested_tensor[0] == scalar
    assert torch.equal(result[0], expected)


def test_ge_scalar(nested_tensor):
    scalar = 6.0
    result = nested_tensor.ge(scalar)
    expected = nested_tensor[0] >= scalar
    assert torch.equal(result[0], expected)


def test_eq_tensor_manual(nested_tensor):
    other = torch.nested.nested_tensor([torch.arange(12).reshape(2, 6)], dtype=torch.float, device="hpu")
    result = torch.nested.nested_tensor(
        [t1.eq(t2) for t1, t2 in zip(nested_tensor.unbind(), other.unbind(), strict=False)]
    )
    expected = nested_tensor[0] == other[0]
    assert torch.equal(result[0], expected)


def test_view(nested_tensor):
    result = nested_tensor.view([1, 2, 6])
    flat_result = result[0].reshape(2, 6)
    assert torch.allclose(flat_result, nested_tensor[0])


def test_copy_(nested_tensor):
    target = torch.nested.nested_tensor([torch.zeros(2, 6, device="hpu")], dtype=torch.float, device="hpu")
    target.copy_(nested_tensor)
    assert torch.allclose(target[0], nested_tensor[0])


def test_normal_(nested_tensor):
    # In-place fill with normal distribution
    nested_tensor.normal_(mean=0.0, std=1.0)

    flat = nested_tensor[0]  # Access underlying Tensor
    mean = flat.mean().item()
    std = flat.std().item()

    # Check if distribution looks reasonable
    assert abs(mean) < 1.0, f"Mean too far from 0: {mean}"
    assert 0.5 < std < 2.0, f"Std deviation out of expected range: {std}"


def test_alias_op(nested_tensor):
    result = torch.ops.aten.alias.default(nested_tensor)
    assert torch.allclose(result[0], nested_tensor[0])
    assert result[0].storage().data_ptr() == nested_tensor[0].storage().data_ptr()


def test_unbind(nested_tensor):
    result = nested_tensor.unbind(dim=0)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert torch.allclose(result[0], nested_tensor[0])
