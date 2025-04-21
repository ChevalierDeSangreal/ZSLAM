import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs import *
from model import *
import time
import argparse

def get_args():
	parser = argparse.ArgumentParser(description="APG Policy")
	
	parser.add_argument("--task", type=str, default="2DRotMov", help="The name of the task.")
	parser.add_argument("--experiment_name", type=str, default="Ver0", help="Name of the experiment to run or load.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed. Overrides config file if provided.")
	parser.add_argument("--device", type=str, default="cuda:0", help="The device")
	
	# train setting
	parser.add_argument("--learning_rate", type=float, default=5.6e-6, help="The learning rate of the optimizer")
	parser.add_argument("--batch_size", type=int, default=1024, help="Batch size of training. Notice that batch_size should be equal to num_envs")
	parser.add_argument("--num_worker", type=int, default=4, help="Number of workers for data loading")
	parser.add_argument("--num_epoch", type=int, default=400900, help="Number of epochs")
	parser.add_argument("--len_sample", type=int, default=60, help="Length of a sample")
	parser.add_argument("--slide_size", type=int, default=20, help="Size of GRU input window")
	
	# model setting
	parser.add_argument("--param_save_name", type=str, default='movingVer0.pth', help="The path to save model parameters")
	parser.add_argument("--param_load_name", type=str, default='movingVer0.pth', help="The path to load model parameters")
	
	args = parser.parse_args()

	
	return args

def get_time():
	timestamp = time.time()
	dt_object_utc = datetime.utcfromtimestamp(timestamp)
	target_timezone = pytz.timezone("Asia/Shanghai")
	dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)
	return dt_object_local.strftime("%Y-%m-%d %H:%M:%S %Z")

@torch.no_grad()
def visualize_predictions(model, envs, device, save_dir, steps=60):
    envs.reset()
    h0 = None

    step_to_visualize = 50  # 只在第3步进行可视化（step从0开始）

    for step in range(steps):
        step_output = envs.step()
        input_tmp = torch.cat((step_output["image"], step_output["agent_pos_encode"]), dim=1)
        gt = step_output["gt"].float()

        output, h0 = model(input_tmp, step_output["gt_position_encode"], h0)
        h0[:, step_output["idx_reset"]] = 0

        if step == step_to_visualize:
            pred = output.detach().cpu().numpy()
            gt = gt.cpu().numpy()

            b_idx = 0  # 可视化第一个样本
            pred_img = pred[b_idx].reshape(50, 50)
            gt_img = gt[b_idx].reshape(50, 50)

            # 保存 gt 图像
            plt.figure(figsize=(5, 5))
            plt.imshow(gt_img, cmap='viridis', vmin=0, vmax=1)
            plt.title(f"GT - Step {step_to_visualize}")
            plt.colorbar()
            plt.axis('off')
            gt_path = os.path.join(save_dir, f"gt_step{step_to_visualize}.png")
            plt.savefig(gt_path)
            plt.close()
            print(f"Saved GT image to: {gt_path}")

            # 保存 pred 图像
            plt.figure(figsize=(5, 5))
            plt.imshow(pred_img, cmap='viridis', vmin=0, vmax=1)
            plt.title(f"Prediction - Step {step_to_visualize}")
            plt.colorbar()
            plt.axis('off')
            pred_path = os.path.join(save_dir, f"pred_step{step_to_visualize}.png")
            plt.savefig(pred_path)
            plt.close()
            print(f"Saved Prediction image to: {pred_path}")

            break  # 可视化完毕后退出

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用 GPU，确保所有 CUDA 设备的随机性固定
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 可能会降低某些情况下的性能，但保证了可复现性

if __name__ == "__main__":
	args = get_args()
	set_seed(args.seed)
	device = args.device

	model_path = os.path.join('/home/wangzimo/VTT/ZSLAM/param/', args.param_load_name)
	model = ZSLAModelVer1(input_dim=68, hidden_dim=64, output_dim=8100, device=device)
	model.load_model(path=model_path, device=device)
	model.eval()

	envs = EnvMove(batch_size=args.batch_size, device=device)

	save_dir = os.path.join('/home/wangzimo/VTT/ZSLAM/output/', "tmpvisl")
	os.makedirs(save_dir, exist_ok=True)

	visualize_predictions(model, envs, device, save_dir, steps=args.len_sample)
	print("Visualization complete.")
