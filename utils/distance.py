import torch

def setwise_distance(a, b=None):
    if b is None:
        b = a
    dist = torch.cdist(a,b,p=2)
    dist = torch.exp(dist)
    return dist
    # return torch.pow((a.unsqueeze(dim=1) - b.unsqueeze(dim=0)), 2.0).sum(dim=-1)