import torch

def single_theta_to_rotation_matrix(theta, device="cpu"):
    theta = torch.tensor(theta)
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)],
    ]).to(device)

def single_theta_to_orientation_vector(theta, device="cpu"):
    theta = torch.tensor(theta)
    return torch.tensor([
        torch.cos(theta),
        torch.sin(theta),
    ]).to(device)

def theta_to_orientation_vector(theta):
    ans = torch.zeros((*theta.shape[:-1], 2), dtype=theta.dtype, device=theta.device)
    ans[..., 0] = torch.cos(theta[..., 0])
    ans[..., 1] = torch.sin(theta[..., 0])
    return ans