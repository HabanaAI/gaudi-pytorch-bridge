import torch

if __name__ == "__main__":
    dtype = torch.float
    """
    device = torch.device("hpu")
    weights = torch.tensor([0, 10, 3, 0], dtype=dtype, device=device)
    wt_h = weights.to(device)
    out = torch.multinomial(wt_h, 2)
    #out = torch.multinomial(weights, 2)
    print(out.cpu())
    """
    device = torch.device("cpu")
    probs = torch.ones((5, 8), dtype=dtype, device=device)
    results = torch.multinomial(probs, 4, replacement=True)
    print("[pyTest] CPU Output:" + str(results.cpu()))

    device = torch.device("hpu")
    hpu_probs = probs.to(device)
    results = torch.multinomial(hpu_probs, 4, True)
    print("[pyTest] HPU Output:" + str(results.cpu()))
