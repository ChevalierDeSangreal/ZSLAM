import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs import EnvMove  # 请确保 env_move 模块在 Python 路径下
from cfg import EnvMoveCfg

def visualize_env_map_and_trajectory(free_points, obstacle_points, safe_points, trajectory, output_path, desired_pos, visible_points=None, is_legend=False):
    """
    绘制环境的自由区域、障碍物、安全区域以及轨迹。
    """
    plt.figure(figsize=(8, 8))
    # 绘制自由点（蓝色）
    if free_points.shape[0] > 0:
        plt.scatter(free_points[:, 0], free_points[:, 1], s=1, c='blue', label='Free Points')
    # 绘制安全区域点（青色）
    if safe_points.shape[0] > 0:
        plt.scatter(safe_points[:, 0], safe_points[:, 1], s=1, c='cyan', label='Safe Points')

    if visible_points is not None:
    
        if visible_points.shape[0] > 0:
            plt.scatter(visible_points[:, 0], visible_points[:, 1], s=1, c='lightcyan', label='Visible Points')
    # 绘制障碍点（红色）
    if obstacle_points.shape[0] > 0:
        plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], s=1, c='red', label='Obstacle Points')

    # 绘制轨迹（黑色折线）
    if len(trajectory) > 0:
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], c='black', linewidth=1.0, label='Agent Trajectory')
        # 轨迹起点（绿色）
        plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=50, marker='o', label='Start')
        # 轨迹终点（黄色）
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='yellow', s=50, marker='X', label='End')

    plt.scatter(desired_pos[0], desired_pos[1], c='purple', s=50, marker='X', label='Desired Pos')

    plt.title("Map Grid and Trajectory Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    if is_legend:
        plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    # 保存图像
    save_path = os.path.join(output_path, "map_trajectory_visualization.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.close()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用 GPU，确保所有 CUDA 设备的随机性固定
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 可能会降低某些情况下的性能，但保证了可复现性


if __name__ == '__main__':
    # set_seed(43)
    output_path = "./output/tmp_map4"
    os.makedirs(output_path, exist_ok=True)
    
    # 初始化环境
    batch_size = 2
    resolution_ratio = 0
    env = EnvMove(batch_size=batch_size, resolution_ratio=resolution_ratio, device="cuda:0")
    
    # 获取环境网格数据
    free_points = env.points_no_obstacle.cpu().numpy()  # 自由网格点
    obstacle_points = env.points_obstacle.cpu().numpy()  # 障碍点
    safe_points = env.points_safe.cpu().numpy()  # 安全区域点
    print(free_points.shape, obstacle_points.shape, safe_points.shape)

    # 轨迹记录
    env.reset()
    list_pos = []

    timer = 0
    while True:
        timer += 1
        idx_reset, desired_pos = env.step()
        pos = env.agent.pos[0].cpu().numpy()
        vel = env.agent.vel[0].cpu().numpy()
        acc = env.agent.acc[0].cpu().numpy()

        #imgs, _ = env.get_images(env.w_gt)
        #print(imgs.shape)
        print(f"Step: {timer}, Pos: {pos}, Vel: {vel}, Acc: {acc}")
        # print(idx_reset)
        if 0 in idx_reset or timer > 500:
            print(f"Reset at {timer}")
            break
        list_pos.append(env.agent.pos[0].cpu().numpy())  # 记录轨迹点
    # desired_pos = env.agent.desired_pos[0].cpu().numpy()
    # 可视化地图和轨迹
    v_point = env.points_all[env.points_visible[0]].cpu().numpy()
    print(v_point.shape)
    visualize_env_map_and_trajectory(free_points, obstacle_points, safe_points, list_pos, output_path, desired_pos[0].cpu().numpy(), visible_points=v_point, is_legend=False)
    # v_map = env.grid_mask_visible[0].T
    # print(.shape)
    # plt.imshow(v_map.cpu().numpy(), cmap="gray", interpolation="nearest")
    # plt.colorbar()
    # plt.title("Boolean Tensor Visualization")
    # plt.show()


