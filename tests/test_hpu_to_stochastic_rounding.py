import numpy as np
<<<<<<< HEAD:tests/test_hpu_to_stochastic_rounding.py
from test_utils import env_var_in_scope, is_gaudi1

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")
=======
import pytest
import torch
from test_utils import env_var_in_scope, is_gaudi2
>>>>>>> 1c8e3092c... [SW-140881] python tests for PT, part 6:tests/pytest_working/test_hpu_to_stochastic_rounding.py

""" torch.fp8r152 dtype has very low precision and stochastic rounding
helps to assure statistically high precision.

E.g. 14.0 and 16.0 are two consecutive fp8 numbers.
Normally, casting 15.1 to fp8 would always give 16.0, which causes a bias.
With stochastic rounding enabled it rounds up or down with the probability
proportional to the distance between the casting number and two closests fp8 numbers.
So effectively, expected value of such casting is the input number itself. """

<<<<<<< HEAD:tests/test_hpu_to_stochastic_rounding.py
@pytest.mark.skipif(is_gaudi1(), reason="G1 doesn't support fp8")
=======

@pytest.mark.skipif(not is_gaudi2())
>>>>>>> 1c8e3092c... [SW-140881] python tests for PT, part 6:tests/pytest_working/test_hpu_to_stochastic_rounding.py
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hpu_to_stochastic_rounding(dtype):
    with env_var_in_scope({"PT_ENABLE_FP8_CAST_STOCHASTIC_ROUNDING": "true"}):
        numbers = np.random.uniform(low=-10000.0, high=10000.0, size=(20,))
        for number in numbers:
            input = torch.full((20, 20), number, dtype=dtype, device="hpu")
            fp8 = input.to(torch.fp8r152)
            fp8_as_float = fp8.float()
            mean = torch.mean(fp8_as_float)
            mean.cpu().numpy() - number
            # TODO choose better metric for comparison
            assert np.allclose(mean.cpu().numpy(), number, atol=0.0, rtol=0.07)
