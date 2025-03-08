import torch

a = torch.tensor([[[1., 1.], [2., 3.]], [[1., 1.], [2., 3.]], [[1., 1.], [2., 3.]]]).unsqueeze(0)
p = torch.tensor([[0., 0.], [0, 0]]).unsqueeze(1)
d = a[..., 1, :] - a[..., 0, :]
r = torch.tensor(1.).unsqueeze(-1).unsqueeze(-1)
f0 = a[..., 0, :] - p
f1 = a[..., 1, :] - p
s = torch.abs((f0[..., 0] * f1[..., 1] - f0[..., 1] * f1[..., 0])).unsqueeze(-1)
h = s / torch.norm(d, dim=-1, keepdim=True)
sign_mask = ((d * f1).sum(dim=-1, keepdim=True) >= 0) & ((-d * f0).sum(dim=-1, keepdim=True) >= 0)
print(h, h.shape)
print(torch.sqrt(torch.tensor(0.2)))
print(sign_mask.shape)
sign = (sign_mask & (h - r <= 0))
print(r.shape)