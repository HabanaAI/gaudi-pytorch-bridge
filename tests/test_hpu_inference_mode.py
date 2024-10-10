import torch
import torch_hpu


# Tests based on https://github.com/pytorch/rfcs/pull/17/files
def test_hpu_inference_mode():
    def check(tensor):
        try:
            tensor._version
        except:  # noqa
            assert tensor.is_inference() is True
        else:
            assert False and "Able to access version counter"

    torch_hpu.is_available()
    hpu = torch.device("hpu")
    a_tensor = torch.randn(4, 4, 64, device=hpu)
    a_tensor_vc = a_tensor._version

    assert a_tensor.is_inference() is False
    b_tensor = a_tensor.view(-1)
    b_tensor_vc = b_tensor._version
    assert b_tensor.is_inference() is False and a_tensor_vc == b_tensor_vc
    b_tensor += 1
    b_tensor_vc_updated = b_tensor._version
    assert b_tensor.is_inference() is False and (b_tensor_vc_updated == b_tensor_vc + 1)
    c_tensor = a_tensor * 2
    c_tensor_vc = c_tensor._version
    assert c_tensor.is_inference() is False and a_tensor_vc == c_tensor_vc

    with torch.inference_mode():
        # Inplace Operation on Normal tensor.
        b_tensor.add_(2)
        b_tensor_vc_updated_i = b_tensor._version
        assert b_tensor.is_inference() is False and (b_tensor_vc_updated_i == b_tensor_vc_updated + 1)

        # Inplace Operation on Inference tensor.
        k_tensor = torch.randn(4, 4, 64, device=hpu)
        check(k_tensor)

        k_tensor.add_(1)
        check(k_tensor)

        # View Op on Normal tensor.
        l_tensor = b_tensor.view(-1)
        assert l_tensor.is_inference() is False and b_tensor_vc_updated_i == l_tensor._version

        # View Op on Inference tensor.
        m_tensor = torch.randn(4, 4, 64, device=hpu)
        n_tensor = m_tensor.view(-1)
        check(n_tensor)

        # Functional Op on Normal tensor.
        o_tensor = b_tensor.mul(2)
        check(o_tensor)

        # Functional Op on Inference tensor.
        p_tensor = m_tensor.mul(2)
        check(p_tensor)

    # Inplace Op on Inference tensor outside of inference_mode
    try:
        p_tensor.add_(1)
    except:  # noqa
        assert p_tensor.is_inference() is True
    else:
        assert False and "Able to do inplace op on inference tensor outside on inference mode"

    # View Op on Inference tensor outside of inference_mode
    r_tensor = p_tensor.view(-1)
    check(r_tensor)
