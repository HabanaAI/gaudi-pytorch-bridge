import pytest
import torch
import torch.nn as nn
from test_utils import evaluate_fwd_kernel

grid_params_list = [
    # op params dict
    (
        {
            "input": torch.randn(1, 1, 2, 2),
            "grid": torch.randn(1, 3, 3, 2),
            "align_corners": True,
        }
    ),
    (
        {
            "input": torch.randn(1, 1, 2, 2),
            "grid": torch.randn(1, 2, 2, 2),
            "align_corners": True,
        }
    ),
    (
        {
            "input": torch.randn(1, 1, 2, 2),
            "grid": torch.randn(1, 3, 3, 2),
            "align_corners": False,
        }
    ),
    (
        {
            "input": torch.randn(1, 1, 2, 2),
            "grid": torch.randn(1, 2, 2, 2),
            "align_corners": False,
        }
    ),
    (
        {
            "input": torch.randn(2, 3, 4, 4),
            "grid": torch.randn(2, 10, 10, 2),
            "align_corners": True,
        }
    ),
    (
        {
            "input": torch.randn(2, 3, 4, 4),
            "grid": torch.randn(2, 10, 10, 2),
            "align_corners": False,
        }
    ),
]


@pytest.mark.parametrize("kernel_params_fwd", grid_params_list)
def test_hpu_grid_op(kernel_params_fwd):
    evaluate_fwd_kernel(kernel=nn.functional.grid_sample, kernel_params=kernel_params_fwd)
