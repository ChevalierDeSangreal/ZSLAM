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
	parser = argparse.ArgumentParser(description="APG Policy")
	
	parser.add_argument("--task", type=str, default="2DRot", help="The name of the task.")
	parser.add_argument("--experiment_name", type=str, default="Ver0", help="Name of the experiment to run or load.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed. Overrides config file if provided.")
	parser.add_argument("--device", type=str, default="cuda:0", help="The device")
	
	# train setting
	parser.add_argument("--learning_rate", type=float, default=5.6e-6, help="The learning rate of the optimizer")
	parser.add_argument("--batch_size", type=int, default=2, help="Batch size of training. Notice that batch_size should be equal to num_envs")
	parser.add_argument("--num_worker", type=int, default=4, help="Number of workers for data loading")
	parser.add_argument("--num_epoch", type=int, default=40900, help="Number of epochs")
	parser.add_argument("--len_sample", type=int, default=20, help="Length of a sample")
	parser.add_argument("--slide_size", type=int, default=10, help="Size of GRU input window")
	
	# model setting
	parser.add_argument("--param_save_path", type=str, default='/home/wangzimo/VTT/ZSLAM/param/twodrotVer0.pth', help="The path to save model parameters")
	parser.add_argument("--param_load_path", type=str, default='/home/wangzimo/VTT/ZSLAM/param_saved/twodrotVer0_5e5_12k.pth', help="The path to load model parameters")
	
	args = parser.parse_args()

	
	return args

def get_time():

	timestamp = time.time()  # 替换为您的时间戳

	# 将时间戳转换为datetime对象
	dt_object_utc = datetime.utcfromtimestamp(timestamp)

	# 指定目标时区（例如"Asia/Shanghai"）
	target_timezone = pytz.timezone("Asia/Shanghai")
	dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)

	# 将datetime对象格式化为字符串
	formatted_time_local = dt_object_local.strftime("%Y-%m-%d %H:%M:%S %Z")

	return formatted_time_local

def dump_yaml(file_path, data):
    """将字典数据保存到 YAML 文件中"""
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)

if __name__ == "__main__":
	torch.autograd.set_detect_anomaly(True)
	args = get_args()
	run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{get_time()}"

	# 保存参数文件
	log_dir = os.path.join('/home/wangzimo/VTT/ZSLAM/runs/', run_name, "env.yaml")
	params = {
		k: v for k, v in args._get_kwargs()
	}
	save_dir = os.path.dirname(log_dir)
	os.makedirs(save_dir, exist_ok=True)
	dump_yaml(log_dir, params)

	writer = SummaryWriter(f"/home/wangzimo/VTT/ZSLAM/runs/{run_name}")

	device = args.device
	print("using device:", device)

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	envs = TwoDEnv(num_envs=args.batch_size, device=args.device)

	model = ZSLAModel(input_dim=64+2, hidden_dim=64, output_dim=3, device=device)
	# model.load_model(path=args.param_load_path, device=device)

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
	criterion = nn.CrossEntropyLoss()
	
	for epoch in range(args.num_epoch):
		print(f"Epoch {epoch} begin...")
		optimizer.zero_grad()

		h0 = None
		envs.reset()
		sum_loss = 0

		for step in range(args.len_sample):

			depth_obs, quad_angle_enc, training_points, gt_labels = envs.step()

			gt_labels = gt_labels + 1

			input_tmp = torch.cat((depth_obs, quad_angle_enc), dim=1)
			output, h0 = model(input_tmp, training_points, h0)
			# print(output[0], gt_labels[0])
			loss = criterion(output, gt_labels)


			h0 = h0.clone()
			sum_loss += loss

		loss.backward(retain_graph=True)
		optimizer.step()
		
		ave_loss = sum_loss / args.len_sample
		print("Ave Loss", ave_loss)
		writer.add_scalar('Loss', ave_loss.item(), epoch)

		if not (epoch % 200) and epoch:
			print("Saving Model...")
			model.save_model(args.param_save_path)

	writer.close()
	print("Training Complete!")