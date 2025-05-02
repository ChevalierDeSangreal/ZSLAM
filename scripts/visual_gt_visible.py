import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random

# 将项目根目录加入路径
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)
from envs import EnvMove
from cfg import EnvMoveCfg
from model import ZSLAModelVer2


def visualize_env_map(free_points, obstacle_points, safe_points, output_path,
                      desired_pos, agent_pos, local_gt_coord):
    """
    绘制环境的自由区域、障碍物、安全区域、目标点、agent 位置，以及 local ground truth 采样点。
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(8, 8))

    if free_points.shape[0] > 0:
        plt.scatter(free_points[:, 0], free_points[:, 1], s=1, c='blue', label='Free Points')
    if obstacle_points.shape[0] > 0:
        plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], s=1, c='red', label='Obstacle Points')
    if safe_points.shape[0] > 0:
        plt.scatter(safe_points[:, 0], safe_points[:, 1], s=1, c='cyan', label='Safe Points')

    # 目标位置
    plt.scatter(desired_pos[0], desired_pos[1], c='purple', s=50, marker='X', label='Desired Pos')
    # agent 当前位置
    plt.scatter(agent_pos[0], agent_pos[1], c='orange', s=30, marker='o', label='Agent Pos')
    # local ground truth 采样点
    if local_gt_coord.shape[0] > 0:
        plt.scatter(local_gt_coord[:, 0], local_gt_coord[:, 1], c='green', s=10, label='Local GT Coord')

    plt.title("Map with Agent and Local GT")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    save_path = os.path.join(output_path, "map.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Map saved to: {save_path}")
    plt.close()


def visualize_visibility_map(grid_mask_visible, grid_map, output_path, filename="visibility_map.png"):
    """
    可视化所有点的可见性。可见点为绿色，不可见点为灰色。
    grid_mask_visible: Tensor of shape (B, W, H)
    grid_map: Tensor of shape (B, W, H, 2)
    """
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(8, 8))

    # print("grid_mask_visible shape:", grid_mask_visible.shape)
    # print("grid_map shape:", grid_map.shape)
    # 仅绘制第 0 个 batch
    visible_mask = grid_mask_visible[0].cpu().numpy()  # (W, H)
    coords = grid_map.cpu().numpy()                # (W, H, 2)

    W, H = visible_mask.shape
    coords_flat = coords.reshape(-1, 2)          # (W*H, 2)
    visibility_flat = visible_mask.reshape(-1)   # (W*H,)

    # 分别绘制可见与不可见点
    visible_coords = coords_flat[visibility_flat == 1]
    invisible_coords = coords_flat[visibility_flat == 0]

    if invisible_coords.shape[0] > 0:
        plt.scatter(invisible_coords[:, 0], invisible_coords[:, 1], s=1, c='lightgray', label='Invisible')
    if visible_coords.shape[0] > 0:
        plt.scatter(visible_coords[:, 0], visible_coords[:, 1], s=1, c='green', label='Visible')

    plt.title("Grid Point Visibility")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visibility map saved to: {save_path}")
    plt.close()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 可选：设置随机种子以复现实验
    set_seed(4117852)

    # 输出路径
    output_path = os.path.join(base_path, "output")
    os.makedirs(output_path, exist_ok=True)

    # 设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 加载环境
    batch_size = 2
    env = EnvMove(batch_size=batch_size, resolution_ratio=0, device=device)

    # 加载模型
    model = ZSLAModelVer2(image_dim=512, hidden_dim=256, query_num=10, num_classes=2, device=device)
    model_load_path = os.path.join(base_path, "param", "movingVer0.pth")
    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"Model file not found at {model_load_path}")
    model.load_model(path=model_load_path, device=device)
    model.eval()

    # 获取地图点集
    free_points = env.points_no_obstacle.cpu().numpy()
    obstacle_points = env.points_obstacle.cpu().numpy()
    safe_points = env.points_safe.cpu().numpy()

    # 环境 reset 并 step
    env.reset()
    step_output = env.step()

    # 采样数据
    desired_pos = env.get_desired_pos()[0].cpu().numpy()
    agent_pos = env.agent.pos[0].cpu().numpy()
    local_gt_coord = step_output["gt"]["local_gt_coord"][0].cpu().numpy()

    # 模型输入准备
    image = step_output["image"][0].unsqueeze(0).to(device)
    agent_pos_enc = step_output["agent_pos_encode"][0].unsqueeze(0).to(device)
    gt = step_output["gt"]
    local_query_encode = gt["local_query_encode"][0].unsqueeze(0).to(device)
    global_query_encode = gt["global_query_encode"][0].unsqueeze(0).to(device)

    # 模型推理，获取预测距离
    with torch.no_grad():
        output_local_distance, _, _ = model(image, agent_pos_enc, 
                                           local_query_encode, global_query_encode)
    pred_distances = output_local_distance[0].cpu().numpy()

    # 打印信息
    print("local gt distance:", gt["local_gt_distance"][0])
    print("pred_distances:", pred_distances)

    # 绘制并保存地图及 GT
    visualize_env_map(free_points, obstacle_points, safe_points,
                      output_path, desired_pos, agent_pos, local_gt_coord)

    # 可视化可见性信息
    visibility_mask = env.grid_mask_visible.cpu()
    grid_map = env.grid_map.cpu()
    visualize_visibility_map(visibility_mask, grid_map, output_path)
