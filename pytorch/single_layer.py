import torch

def activation(x):
    return 1/ (1 + torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1,5))
weights = torch.randn_like(features)
bias = torch.randn((1,1))

y = activation(torch.sum(features * weights) + bias)
y = activation((featurs * weights).sum() + bias)
y = activation(torch.mm(features, weights.view(5,1)) + bias)

print(y)