from map import Map
from utils.geometry import get_circle_points, get_line_points, transformation
import torch

def draw_new_triangle(grid, p0, p1, p2, ratio=0.02):
    grid_ = torch.empty_like(grid)
    grid_.copy_(grid)
    device = grid.device
    H, W = grid.shape[0], grid.shape[1]
    center = torch.tensor([H//2, W//2], dtype=torch.int32, device=device)
    line = torch.tensor([
        [p0, p1],
        [p1, p2],
        [p2, p0]
    ]).to(device) / ratio
    points = get_line_points(lines=line)
    points = transformation(points, center)

    valid = (points[:, 0] >= 0) & (points[:, 0] < H) & (points[:, 1] >= 0) & (points[:, 1] < W)
    points = points[valid]
    grid_[points[:, 0], points[:, 1]] = True

    return grid_


def draw_new_circle(grid, x, y, r, ratio=0.02):
    grid_ = torch.empty_like(grid)
    grid_.copy_(grid)
    device = grid.device
    H, W = grid.shape[0], grid.shape[1]
    center = torch.tensor([H//2, W//2], dtype=torch.int32, device=device)
    circle_center = torch.tensor([[x, y]], device=device) / ratio
    circle_radius = torch.tensor([[r]], device=device).to(device) / ratio
    points = get_circle_points(circle_center, circle_radius)
    points = transformation(points, center)

    valid = (points[:, 0] >= 0) & (points[:, 0] < H) & (points[:, 1] >= 0) & (points[:, 1] < W)
    points = points[valid]
    grid_[points[:, 0], points[:, 1]] = True

    return grid_

def draw_grid(map:Map, ratio=0.02, device="cpu"):
    H, W = int(map.height/ratio), int(map.width/ratio)
    grid = torch.zeros((H, W), device=device, dtype=torch.bool)
    center = torch.tensor([H//2, W//2], dtype=torch.int32, device=device)

    points = None

    if len(map.circle_center_array) != 0:
        circle_center = torch.stack(map.circle_center_array).to(device) / ratio
        circle_radius = torch.stack(map.circle_radius_array).to(device) / ratio
        points = get_circle_points(circle_center, circle_radius)

    if len(map.line_array) != 0:
        line = torch.stack(map.line_array).to(device) / ratio
        line_points = get_line_points(lines=line)
        if points is not None:
            points = torch.cat([points, line_points], dim=0)
        else:
            points = line_points
    
    if points is not None:
        points = transformation(points, center)
        valid = (points[:, 0] >= 0) & (points[:, 0] < H) & (points[:, 1] >= 0) & (points[:, 1] < W)
        points = points[valid]
        grid[points[:, 0], points[:, 1]] = True
    
    return grid