import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append('/home/wangzimo/VTT/ZSLAM')

from envs import EnvMove  # 请确保 env_move 模块在 Python 路径下
from cfg import EnvMoveCfg

def visualize_env_map():
    """
    测试脚本流程：
      1. 实例化 EnvMove，使用类内部生成的地图（包括障碍物和安全区域）。
      2. 从 EnvMove 中提取障碍点（points_obstacle）与安全区域（safe_set）的网格点。
      3. 利用 matplotlib 绘制障碍点（红色）与安全点（青色）的散点图。
    """
    output_path="/home/wangzimo/VTT/ZSLAM/output"
    # 同时在 init_grid 中计算了 points_obstacle 与 safe_set
    batch_size = 2
    resolution_ratio = 0
    env = EnvMove(batch_size=batch_size, resolution_ratio=resolution_ratio, device="cpu")
    
    # 从 EnvMove 中获取障碍点和安全区域的网格点
    # 这两个变量在 env.init_grid() 中已由 get_map_grid 计算，并经过 transformation_back 转换
    free_points = env.points_no_obstacle.cpu().numpy()              # 自由网格点
    obstacle_points = env.points_obstacle.cpu().numpy()
    safe_points = env.safe_set.cpu().numpy()
    print(free_points.shape)
    plt.figure(figsize=(8, 8))
    # 绘制自由点（蓝色）
    if free_points.shape[0] > 0:
        plt.scatter(free_points[:, 0], free_points[:, 1], s=1, c='blue', label='Free Points')
    # 绘制障碍点（红色）
    if obstacle_points.shape[0] > 0:
        plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], s=1, c='red', label='Obstacle Points')
    # 绘制安全区域点（青色）
    if safe_points.shape[0] > 0:
        plt.scatter(safe_points[:, 0], safe_points[:, 1], s=1, c='cyan', label='Safe Points')
    
    plt.title("Map Grid Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    # 保存图像
    save_path = os.path.join(output_path, "map_visualization.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.close()

if __name__ == '__main__':
    visualize_env_map()
