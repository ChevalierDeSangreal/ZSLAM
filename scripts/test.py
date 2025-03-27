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
                        default="/home/wangzimo/VTT/ZSLAM/param_saved/twodrotVer0_5e4_better.pth",
                        help="加载模型参数的路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--batch_size", type=int, default=1024, help="测试时的批量大小")
    parser.add_argument("--test_steps", nargs="+", type=int,
                        # default=[5, 7, 9, 10, 15, 20, 25, 27, 29, 30, 33, 35, 37],
                        # default=[1, 2, 5, 7, 9, 10, 15, 20, 25],
                        default=[10, 30, 50, 70, 90, 110, 130, 150],
                        help="测试时需要评估的步数列表，要求单调递增")
    parser.add_argument("--output_dir", type=str,
                        default="/home/wangzimo/VTT/ZSLAM/test_runs",
                        help="测试结果输出目录")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="每个测试步数生成的样本数量")
    args = parser.parse_args()
    return args


def visualize_samples_on_single_map(env, training_points, gt_labels, predictions, depth_obs, cam_angle, fov,
                                    step_count, env_idx=0, output_dir="."):
    """
    将所有样本的结果可视化在同一张地图上：
      - 绘制一次地图背景（安全区域和所有圆形组件）。
      - 对于每个样本，根据 ground truth 和预测结果：
          * 预测正确：使用不同的 ground truth 对应不同的颜色；
          * 预测错误：显示红色；
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
    
    # 定义颜色列表，用于预测正确时不同的 ground truth
    color_list = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    # 用于生成图例：记录预测正确时每个 ground truth 的颜色
    correct_gt_colors = {}
    incorrect_found = False
    
    # 遍历所有样本，根据预测正确性选用颜色
    for i in range(num_samples):
        # 训练点为 (cos(theta), sin(theta), distance)
        tp = training_points[i, env_idx].cpu().numpy()  
        x_tp = tp[0] * tp[2]
        y_tp = tp[1] * tp[2]
        gt_val = gt_labels[i, env_idx].item()
        pred_val = predictions[i, env_idx].item()
        
        if pred_val == gt_val:
            # 根据 ground truth 选择颜色
            color = color_list[int(gt_val) % len(color_list)]
            correct_gt_colors[gt_val] = color
        else:
            color = "red"
            incorrect_found = True
        
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
    
    # 构造图例：对于预测正确的样本，每个不同的 ground truth 使用一种颜色；预测错误统一为红色
    from matplotlib.lines import Line2D
    legend_elements = []
    for gt, col in sorted(correct_gt_colors.items()):
        legend_elements.append(Line2D([0], [0], marker='x', color='w', label=f"Correct GT {gt}",
                                      markerfacecolor=col, markersize=10))
    if incorrect_found:
        legend_elements.append(Line2D([0], [0], marker='x', color='w', label='Incorrect',
                                      markerfacecolor='red', markersize=10))
    
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
    h = None
    with torch.no_grad():
        # 在完整的步数递增过程中，按步模拟环境
        while current_step < max_step:
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

            output, h = model(input_flat, training_points_flat, h)

            if (current_step + 1) in test_steps:
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
                                                env_idx=0,
                                                output_dir=args.output_dir)

    # 保存所有指定测试步数下的准确率到 YAML 文件
    os.makedirs(args.output_dir, exist_ok=True)
    yaml_path = os.path.join(args.output_dir, "multisample_one_map_accuracies.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(overall_accuracies, f)
    print(f"Saved accuracies to {yaml_path}")


if __name__ == "__main__":
    main()