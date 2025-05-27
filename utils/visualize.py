import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random


def visualize_env_map(
    free_points, 
    obstacle_points, 
    safe_points, 
    desired_pos, 
    agent_pos, 
    ratio, 
    center, 
    canvas_size=512, 
    device="cpu"):
    """
    绘制环境的自由区域、障碍物、安全区域、目标点、agent 位置。

    """
    def normalize(coords):
        return ((coords  / (2 * ratio * center) + 0.5) * canvas_size).long()

    def draw_points(coords, color):
        coords = normalize(coords)
        mask = (coords[:,0] >= 0) & (coords[:,0] < canvas_size) & \
               (coords[:,1] >= 0) & (coords[:,1] < canvas_size)
        canvas[coords[mask,1], coords[mask,0]] = torch.tensor(color, device=device)
        return
    
    #canvas = torch.full((canvas_size, canvas_size, 3), 255, dtype=torch.uint8, device=device)
    canvas = torch.full((canvas_size, canvas_size, 3), 255, dtype=torch.long, device=device)
    
    draw_points(free_points, [0, 0, 255])      # 蓝色自由区域
    draw_points(obstacle_points, [255, 0, 0])  # 红色障碍物
    draw_points(safe_points, [255, 255, 0])    # 青色安全区

    def draw_special(pos, color, size=5):
        pos = normalize(pos.unsqueeze(0))[0]
        y, x = pos[1].clamp(0, canvas_size-1), pos[0].clamp(0, canvas_size-1)
        canvas[y-size:y+size, x-size:x+size] = torch.tensor(color, device=device)

    
    draw_special(desired_pos, [128, 0, 128], size=4)  # 紫色目标
    draw_special(agent_pos, [0, 165, 255], size=4)    # 橙色Agent
    
    # 返回CPU上的numpy数组 (BGR格式)
    return canvas.cpu().numpy().astype(np.uint8)[..., ::-1]  # RGB->BGR
