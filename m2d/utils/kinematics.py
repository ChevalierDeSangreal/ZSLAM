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

if __name__=="__main__":
    theta = 0.
    #theta = -3.1415926
    print(single_theta_to_orientation_vector(theta=theta))