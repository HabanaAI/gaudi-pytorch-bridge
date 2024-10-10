import pytest
import test_utils
import torch
from test_utils import compare_tensors, env_var_in_scope, is_lazy

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")

try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    raise AssertionError("Could Not import habana_frameworks.torch.core")


test_case_list = [
    # D, H, W, * 2
    [(132, 176, 136), (146, 175, 146)]
]


@pytest.mark.xfail
@pytest.mark.skipif(not is_lazy(), reason="Lazy only test")
@pytest.mark.parametrize("dyn_shape", test_case_list)
def test_hpu_permute(dyn_shape):
    def fun(img, lbl, dev):
        img = img.to(dev)
        lbl = lbl.to(dev)
        img = img.contiguous(memory_format=torch.channels_last_3d)
        lbl = lbl.contiguous(memory_format=torch.channels_last_3d)
        if dev == torch.device("hpu"):
            htcore.mark_step()
        return img, lbl

    shapes = dyn_shape
    with env_var_in_scope({"PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES": 1}):
        for i in range(len(shapes)):
            x, y, z = shapes[i]
            input = torch.randn((1, 4, x, y, z))
            target = torch.randn((1, 1, x, y, z))
            i_c, t_c = fun(input, target, test_utils.cpu)
            i_h, t_h = fun(input, target, test_utils.hpu)
            compare_tensors(i_h, i_c, 1e-3, 1e-3)
            compare_tensors(t_h, t_c, 1e-3, 1e-3)


if __name__ == "__main__":
    test_hpu_permute(*test_case_list[0])
