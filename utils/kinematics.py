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

def batch_theta_to_rotation_matrix(theta):
    theta = torch.as_tensor(theta)  # 确保 theta 是张量
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    R = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=-1),
        torch.stack([sin_theta, cos_theta], dim=-1)
    ], dim=-2)  # 形状变为 (batch_size, 2, 2)
    
    return R

def batch_theta_to_orientation_vector(theta):
    theta = torch.as_tensor(theta)  # 确保 theta 是张量
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    orientation_vectors = torch.stack([cos_theta, sin_theta], dim=-1)  # 形状变为 (batch_size, 2)
    
    return orientation_vectors