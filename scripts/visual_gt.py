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
    print("local_gt_coord shape:", local_gt_coord.shape)
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 可选：设置随机种子以复现实验
    # set_seed(4117852)

    # 输出路径
    output_path = os.path.join(base_path, "output")
    # print(output_path)
    # print(base_path)
    os.makedirs(output_path, exist_ok=True)

    # 设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 加载环境
    batch_size = 2
    env = EnvMove(batch_size=batch_size, resolution_ratio=0, device=device)

    # 加载模型
    model = ZSLAModelVer2(image_dim=512, hidden_dim=256, query_num=10, num_classes=2, device=device)
    model_load_path = os.path.join(base_path, "param", "movingVer0.pth")
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
    desired_pos = env.get_desired_pos()[0].cpu().numpy()         # (2,)
    agent_pos = env.agent.pos[0].cpu().numpy()                   # (2,)
    local_gt_coord = step_output["gt"]["local_gt_coord"][0].cpu().numpy()  # (10,2)

    # 模型输入准备
    image = step_output["image"][0].unsqueeze(0).to(device)
    agent_pos_enc = step_output["agent_pos_encode"][0].unsqueeze(0).to(device)
    gt = step_output["gt"]
    local_query_encode = gt["local_query_encode"][0].unsqueeze(0).to(device)
    global_query_encode = gt["global_query_encode"][0].unsqueeze(0).to(device)

    # 模型推理，获取预测距离
    with torch.no_grad():
        output_local_distance, output_local_class, output_global_exprate = model(image, agent_pos_enc, 
                                           local_query_encode, global_query_encode)
    pred_distances = output_local_distance[0].cpu().numpy()  # (10,)     # (2,)
    # print("image", image[0])
    print("local gt distance", gt["local_gt_distance"][0])
    print("pred_distances:", pred_distances)
    print("local gt coord:", local_gt_coord)
    print("global output", output_global_exprate[0])
    # 绘制并保存
    visualize_env_map(free_points, obstacle_points, safe_points,
                      output_path, desired_pos, agent_pos, local_gt_coord)
