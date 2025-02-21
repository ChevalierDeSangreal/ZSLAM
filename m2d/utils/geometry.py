import torch


def get_line_points(lines):
    x1, y1 = lines[:, 0, 0], lines[:, 0, 1]
    x2, y2 = lines[:, 1, 0], lines[:, 1, 1]
    
    num_points = torch.maximum(torch.abs(x2 - x1), torch.abs(y2 - y1)) + 1
    num_points = num_points.max().to(torch.int32).item()

    weights = torch.linspace(0, 1, num_points, device=lines.device)
    x = x1.unsqueeze(1) + (x2 - x1).unsqueeze(1) * weights
    y = y1.unsqueeze(1) + (y2 - y1).unsqueeze(1) * weights
    points = torch.stack([x, y], dim=-1).view(-1, 2).to(torch.int32)
    return points

def get_circle_points(c, r):
    N = c.shape[0]
    num_samples = int(2 * torch.pi * torch.max(r)) + 4
    theta = torch.linspace(0, 2 * torch.pi, steps=num_samples, device=c.device)

    x = c[:, 0:1] + r * torch.cos(theta)
    y = c[:, 1:2] + r * torch.sin(theta)

    x = torch.round(x).to(torch.int32)
    y = torch.round(y).to(torch.int32)
    points = torch.stack([x, y], dim=-1).view(-1, 2)

    return points

def transformation(coordinates, center):
    x, y = coordinates[:, 0], coordinates[:, 1]
    return torch.stack([-y, x], dim=1) + center

def not_triangle(p0, p1, p2, device="cpu"):
    p0, p1, p2 = torch.tensor(p0, device=device), torch.tensor(p1, device=device), torch.tensor(p2, device=device)
    n0, n1 = p2 - p0, p2 - p1
    sign = n0[0] * n1[1] - n0[1] * n1[0]
    return sign == 0
    