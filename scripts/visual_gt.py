import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs import EnvMove
from cfg import EnvMoveCfg

def visualize_ground_truth(ground_truth: torch.Tensor, 
                           square_size: int, output_path: str = "output"):
    """
    可视化 batch 0 的 ground_truth 数据，并保存到本地图像文件。

    参数:
        ground_truth (torch.Tensor): 可见可达信息，形状为 (B, square_size * square_size)。
        square_size (int): 正方形边长，用于 reshape。
        save_path (str): 图像保存路径。
    """
    # 只处理 batch 0
    print(f"Ground truth shape: {ground_truth.shape}")
    gt_0 = ground_truth[0].reshape(square_size, square_size).cpu().numpy()

    # 设置颜色映射：0->黑色（不可见）、1->绿色（可达）、2->红色（不可达）
    cmap = plt.cm.get_cmap("tab10", 3)
    plt.figure(figsize=(4, 4))
    plt.imshow(gt_0, cmap=cmap, vmin=0, vmax=2)
    plt.title("Ground Truth Visualization (Batch 0)")
    plt.colorbar(ticks=[0, 1, 2], label="Visibility")
    plt.clim(-0.5, 2.5)
    plt.xticks([])
    plt.yticks([])

    save_path = os.path.join(output_path, "ground_truth_vis.png")
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Ground truth visualization saved to {save_path}")

def visualize_env_map_and_trajectory(free_points, obstacle_points, safe_points, trajectory, output_path, desired_pos):
    """
    绘制环境的自由区域、障碍物、安全区域以及轨迹
    """
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(8, 8))

    if free_points.shape[0] > 0:
        plt.scatter(free_points[:, 0], free_points[:, 1], s=1, c='blue', label='Free Points')
    if obstacle_points.shape[0] > 0:
        plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], s=1, c='red', label='Obstacle Points')
    if safe_points.shape[0] > 0:
        plt.scatter(safe_points[:, 0], safe_points[:, 1], s=1, c='cyan', label='Safe Points')

    if len(trajectory) > 0:
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], c='black', linewidth=1.0, label='Agent Trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=50, marker='o', label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='yellow', s=50, marker='X', label='End')

    plt.scatter(desired_pos[0], desired_pos[1], c='purple', s=50, marker='X', label='Desired Pos')

    plt.title("Map and Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    save_path = os.path.join(output_path, "map_with_trajectory.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Map and trajectory saved to: {save_path}")
    plt.close()

def visualize_visible_points_only(visible_points, output_path):
    """
    仅绘制最终可视区域点
    """
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(8, 8))
    if visible_points.shape[0] > 0:
        plt.scatter(visible_points[:, 0], visible_points[:, 1], s=1, c='magenta', label='Visible Points')
    
    plt.title("Visible Area After Last Step")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    save_path = os.path.join(output_path, "visible_area_only.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visible points saved to: {save_path}")
    plt.close()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_visible_mask(mask, output_path, prefix="visible_mask"):
    """
    支持 batch_size ≥ 1 的可视化函数。
    每个 batch 的 mask 会保存为一张单独图片。
    
    参数:
        mask: Tensor[batch_size, W, H]，bool 类型
        output_path: 保存图片的目录
        prefix: 文件名前缀，默认是 'visible_mask'
    """
    mask = mask.cpu().numpy()  # shape: [B, W, H]

    i = 0
    plt.figure(figsize=(6, 6))
    print(f"Mask shape: {mask.shape}")
    plt.imshow(mask[i], cmap='hot', interpolation='nearest')  # 也可使用 'binary' 或 'gray'
    plt.title(f"Visible Mask - Batch {i}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label='Visible (1=True, 0=False)')
    plt.gca().set_aspect('equal', adjustable='box')

    save_path = os.path.join(output_path, f"{prefix}_{i}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visible mask for batch {i} saved to: {save_path}")
    plt.close()

def visualize_visible_mask_on_grid(grid, mask, output_path, prefix="visible_mask"):
    """
    使用传入的 grid (W, H, 2) 对 visible mask 进行可视化。
    
    参数:
        grid: np.ndarray 或 Tensor，形状为 (W, H, 2)，存储每个网格点的 (x, y) 坐标
        mask: Tensor 形状为 (B, W, H)，dtype 为 bool
        output_path: 图片保存目录
        prefix: 文件名前缀，默认 'visible_mask'
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 如果 grid 是 Tensor，则转换成 numpy 数组
    if hasattr(grid, "cpu"):
        grid = grid.cpu().numpy()
    
    # 判断 mask 是否为 batch 格式
    mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else mask
    # 多个 batch 的情况，仅可视化第 0 个 batch
    mask0 = mask_np[0]  # shape: (W, H)
    visible_coords = grid[mask0]  # shape: (n, 2)
    plt.figure(figsize=(6, 6))
    plt.scatter(visible_coords[:, 0], visible_coords[:, 1], s=1, c='magenta', label='Visible Points')
    plt.title("Visible Mask using Grid Coordinates - Batch 0")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    save_path = os.path.join(output_path, f"{prefix}_0.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visible mask for batch 0 saved to: {save_path}")
    plt.close()

if __name__ == '__main__':
    # set_seed(412)
    set_seed(42)
    output_path = "./output/visual_gt"

    batch_size = 2
    resolution_ratio = 0
    env = EnvMove(batch_size=batch_size, resolution_ratio=resolution_ratio, device="cuda:0")

    free_points = env.points_no_obstacle.cpu().numpy()
    obstacle_points = env.points_obstacle.cpu().numpy()
    safe_points = env.points_safe.cpu().numpy()
    grid = env.grid_map.cpu().numpy()  # shape: (W, H, 2)

    env.reset()
    list_pos = [env.agent.pos[0].cpu().numpy()]
    timer = 0
    desired_pos = env.get_desired_pos()
    delta_point_mask_visible, delta_grid_mask_visible = env.get_img_field()  # shape: [batch_size, W, H], dtype=bool
    # visualize_visible_mask_on_grid(grid, delta_grid_mask_visible, output_path)

    visible_points = env.points_visible.cpu().numpy()
    visualize_visible_points_only(visible_points, output_path)
    
    while True:
        timer += 1
        step_output = env.step()
        idx_reset = step_output["idx_reset"]

        pos = env.agent.pos[0].cpu().numpy()
        if timer % 10 == 0:
            print(f"Step {timer}, Pos: {pos}")
        
        if 0 in idx_reset or timer > 150:
            print(f"Reset at step {timer}")
            break
        list_pos.append(pos)

    visible_points = env.points_visible.cpu().numpy()
    desired_pos_np = desired_pos[0].cpu().numpy()

    # 分别保存地图轨迹图 & 可见区域图
    visualize_env_map_and_trajectory(free_points, obstacle_points, safe_points, list_pos, output_path, desired_pos_np)
    
    visible_grid = env.grid_mask_visible.cpu().numpy()
    visualize_visible_mask_on_grid(grid, visible_grid, output_path)

    coord, gt = env.generate_ground_truth(90)
    print(f"Coordinate: {coord[0]}")
    visualize_ground_truth(gt, 90, output_path)

