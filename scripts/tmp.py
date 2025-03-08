import torch

tenor1 = torch.tensor([[1, 2, 3]])
tensor2 = torch.tensor([[4, 5, 6]])
print(tenor1.shape)

print(torch.stack((tenor1, tensor2), dim=0).shape)