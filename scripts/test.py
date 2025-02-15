import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
import yaml
import argparse

import sys
sys.path.append('/home/wangzimo/VTT/ZSLAM')

from envs import *
from model import *


def get_args():
    parser = argparse.ArgumentParser(description="Test ZSLAModel")
    parser.add_argument("--param_load_path", type=str,
                        default="/home/wangzimo/VTT/ZSLAM/param/twodrotVer0.pth",
                        help="加载模型参数的路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--batch_size", type=int, default=2, help="测试时的批量大小")
    parser.add_argument("--test_steps", nargs="+", type=int,
                        # default=[5, 7, 9, 10, 15, 20, 25, 45, 55, 65, 75, 85, 95, 100],
                        default=[5, 7, 9, 10, 15],
                        help="测试时需要评估的步数列表，要求单调递增")
    parser.add_argument("--output_dir", type=str,
                        default="/home/wangzimo/VTT/ZSLAM/test_runs",
                        help="测试结果输出目录")
    parser.add_argument("--num_samples", type=int, default=75,
                        help="每个测试步数生成的样本数量")
    args = parser.parse_args()
    return args


def visualize_samples_on_single_map(env, training_points, gt_labels, predictions, depth_obs, cam_angle, fov,
                                    step_count, env_idx=0, output_dir="."):
    """
    将所有样本的结果可视化在同一张地图上：
      - 绘制一次地图背景（安全区域和所有圆形组件）。
      - 对于每个样本，根据 ground truth 和预测结果：
          * 预测正确：绿色；
          * 预测错误：红色；
      - 将训练点和对应的深度射线击中点均绘制在同一张地图上。
    """
    num_samples = training_points.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    map_r = env.map.map_r
    ax.set_title(f"Step {step_count} - All Samples on One Map (Env {env_idx})", fontsize=16)
    ax.set_xlim(-map_r - 1, map_r + 1)
    ax.set_ylim(-map_r - 1, map_r + 1)
    ax.set_aspect('equal')
    
    # 绘制安全区域
    safe_circle = plt.Circle((0, 0), 0.5, color='gray', fill=False, linestyle='--', label='Safe Zone')
    ax.add_artist(safe_circle)
    
    # 绘制地图上所有圆形组件（地图物体）
    for (cx, cy, r) in env.map.map_componet[env_idx].cpu().numpy():
        circle = plt.Circle((cx, cy), r, color='blue', fill=False)
        ax.add_artist(circle)
    
    # 遍历所有样本，根据预测正确性选用颜色
    for i in range(num_samples):
        tp = training_points[i, env_idx].cpu().numpy()  # (cos(theta), sin(theta), distance)
        x_tp = tp[0] * tp[2]
        y_tp = tp[1] * tp[2]
        gt_val = gt_labels[i, env_idx].item()
        pred_val = predictions[i, env_idx].item()
        
        color = "green" if pred_val == gt_val else "red"
        
        # 绘制训练点
        ax.scatter([x_tp], [y_tp], color=color, s=100, marker='x')
        # 标注样本索引及 GT/Pred 信息
        ax.text(x_tp, y_tp, f"{i}\nGT:{gt_val}\nPred:{pred_val}", color=color, fontsize=10)
        
        # 绘制该样本对应的深度射线击中点
        depth = depth_obs[i, env_idx].cpu().numpy()  # shape: (num_rays,)
        num_rays = len(depth)
        angle = cam_angle[env_idx].item()
        ray_angles = np.linspace(-fov/2, fov/2, num_rays) + angle
        x_rays = depth * np.cos(ray_angles)
        y_rays = depth * np.sin(ray_angles)
        ax.scatter(x_rays, y_rays, color=color, s=10, alpha=0.6)
    
    # 添加图例说明
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', label='Correct', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='x', color='w', label='Incorrect', markerfacecolor='red', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"test_step_{step_count}_all_samples_one_map.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved combined samples visualization to {save_path}")


def main():
    args = get_args()
    device = args.device

    # 固定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 初始化环境与模型，并固定地图
    envs = TwoDEnv(num_envs=args.batch_size, device=device)
    envs.reset()
    fixed_map = envs.map  # 保存固定地图

    # 修改模型输出维度为3
    model = ZSLAModel(input_dim=64+2, hidden_dim=64, output_dim=3, device=device)
    model.load_model(path=args.param_load_path, device=device)
    model.eval()

    test_steps = sorted(args.test_steps)  # 确保步数单调递增
    num_samples = args.num_samples
    overall_accuracies = {}  # 保存指定测试步数下的准确率

    current_step = 0
    max_step = max(test_steps)
    # 在完整的步数递增过程中，按步模拟环境
    while current_step < max_step:
        if (current_step + 1) in test_steps:
            # 在指定步数处采用多样本 step 检查结果
            depth_obs, quad_angle_enc, training_points, gt_labels = envs.step(num_sample=num_samples)
            gt_labels = gt_labels + 1  # 对齐训练时的标签处理
            
            current_step += 1
            num_envs = args.batch_size

            # 扩展无人机朝向编码至多样本维度，并拼接深度观测
            quad_angle_enc_exp = quad_angle_enc.unsqueeze(0).expand(num_samples, -1, -1)
            model_input = torch.cat((depth_obs, quad_angle_enc_exp), dim=-1)
            input_flat = model_input.view(num_samples * num_envs, -1)
            training_points_flat = training_points.view(num_samples * num_envs, -1)

            output, _ = model(input_flat, training_points_flat, None)
            # 修改预测处理逻辑
            output = output.view(num_samples, num_envs, 3)
            predictions = torch.argmax(output, dim=-1)

            # 计算准确率
            total = gt_labels.numel()
            correct = (predictions == gt_labels).sum().item()
            avg_accuracy = correct / total
            overall_accuracies[current_step] = avg_accuracy
            print(f"Step {current_step}: Average Accuracy = {avg_accuracy*100:.2f}%")

            # 对指定环境进行可视化
            cam_angle = envs.quad_state[:, 0]  # 当前无人机朝向
            visualize_samples_on_single_map(env=envs,
                                            training_points=training_points,
                                            gt_labels=gt_labels,
                                            predictions=predictions,
                                            depth_obs=depth_obs,
                                            cam_angle=cam_angle,
                                            fov=envs.fov,
                                            step_count=current_step,
                                            env_idx=1,
                                            output_dir=args.output_dir)
        else:
            # 非测试步数仅采用单样本 step 更新环境
            _ = envs.step(num_sample=1)
            current_step += 1

    # 保存所有指定测试步数下的准确率到 YAML 文件
    os.makedirs(args.output_dir, exist_ok=True)
    yaml_path = os.path.join(args.output_dir, "multisample_one_map_accuracies.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(overall_accuracies, f)
    print(f"Saved accuracies to {yaml_path}")


if __name__ == "__main__":
    main()