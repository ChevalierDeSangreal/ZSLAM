import os
import sys
import random
import time
import math
import argparse
from datetime import datetime

import numpy as np
import torch
import pytz
import matplotlib.pyplot as plt

# 将项目根路径添加到 sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)

from envs import EnvMove
from model import ZSLAModelVer1


def get_args():
    parser = argparse.ArgumentParser(description="APG Policy Visualization")

    # Environment and model settings
    parser.add_argument("--task", type=str, default="2DRotMov", help="The name of the task.")
    parser.add_argument("--experiment_name", type=str, default="Ver0", help="Name of the experiment to run or load.")
    parser.add_argument("--seed", type=int, default=4165782, help="Random seed. Overrides config file if provided.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device")

    # Data and visualization parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for environment.")
    parser.add_argument("--len_sample", type=int, default=60, help="Length of a sample (number of steps).")
    parser.add_argument("--step_to_vis", type=int, default=50, help="Which step index to visualize (0-based).")
    parser.add_argument("--batch_idx", type=int, default=0, help="Which batch element to visualize.")

    # Model checkpoint
    parser.add_argument("--param_load_name", type=str, default="movingVer0.pth", 
                        help="Filename of model parameters under param/ directory.")

    return parser.parse_args()


def get_time_str():
    ts = time.time()
    dt_utc = datetime.utcfromtimestamp(ts)
    tz = pytz.timezone("Asia/Shanghai")
    dt_local = dt_utc.replace(tzinfo=pytz.utc).astimezone(tz)
    return dt_local.strftime("%Y%m%d_%H%M%S")

@torch.no_grad()
def visualize_predictions(model, envs, device, save_dir, step_to_vis, batch_idx):
    # 重置环境
    envs.reset()
    h0 = None

    # 推理每一步
    for step in range(envs.len_sample if hasattr(envs, 'len_sample') else step_to_vis + 1):
        out = envs.step()
        img_enc = out["image"].to(device)
        pos_enc = out["agent_pos_encode"].to(device)
        gt_enc = out["gt"].float().to(device)
        gt_pos_enc = out["gt_position_encode"].to(device)

        inputs = torch.cat((img_enc, pos_enc), dim=1)
        output, h0 = model(inputs, gt_pos_enc, h0)


        # 到达要可视化的步数
        if step == step_to_vis:
            pred = output.detach().argmax(dim=2).cpu().numpy()
            gt   = gt_enc.cpu().numpy()
            # 计算输出维度对应的图像尺寸
            img_size = 40

            # 获取指定样本
            pred_img = pred[batch_idx].reshape(img_size, img_size)
            gt_img   = gt[batch_idx].reshape(img_size, img_size)

            timestamp = get_time_str()
            # 保存并绘制
            for name, data in [("gt", gt_img), ("pred", pred_img)]:
                plt.figure(figsize=(6, 6))
                plt.imshow(data, cmap='viridis', vmin=0, vmax=2)
                plt.title(f"{name.upper()} at step {step}, batch {batch_idx}")
                plt.colorbar()
                plt.axis('off')

                filename = f"{name}.png"
                out_path = os.path.join(save_dir, filename)
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                print(f"Saved {name.upper()} image: {out_path}")
            break


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # 加载模型
    model = ZSLAModelVer1(input_dim=64 + 12 + 4, hidden_dim=64, output_dim=1600, device=device)
    model_path = os.path.join(base_dir, 'param', args.param_load_name)
    model.load_model(path=model_path, device=device)
    model.eval()

    # 创建环境
    envs = EnvMove(batch_size=args.batch_size, device=device)
    # 将 len_sample 保存到 envs 对象，供 visualize 使用
    envs.len_sample = args.len_sample

    # 输出目录
    save_dir = os.path.join(base_dir, 'output', args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # 可视化
    visualize_predictions(model, envs, device, save_dir, 
                          step_to_vis=args.step_to_vis, 
                          batch_idx=args.batch_idx)
    print("Visualization complete.")
