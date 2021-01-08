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
    total_norm = hb_custom_C.fused_norm(norm_list, norm_type)

    max_norm = float(max_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm
