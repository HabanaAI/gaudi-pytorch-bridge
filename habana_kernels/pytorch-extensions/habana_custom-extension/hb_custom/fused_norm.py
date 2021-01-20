import torch

def FusedClipNorm(parameters, max_norm):
    try:
        import hb_custom_C
    except ImportError:
        raise ImportError("Could not import hb_custom_C")

    hpu = torch.device("habana")
    cpu = torch.device("cpu")

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_list = []
    for p in parameters:
        norm_list.append(p.grad.detach())

    norm_type = 2.0
    max_norm_t = (torch.ones((1))*max_norm).to(hpu)
    total_norm = hb_custom_C.fused_norm(norm_list, max_norm_t, norm_type)

    return total_norm
