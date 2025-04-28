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
base_path = '/home/wangzimo/VTT/ZSLAM'
sys.path.append(base_path)

from envs import *
from model import *

"""
相比于train_envmove.py，使用了新讨论的预训练方式，将local和global map feature分开。
使用了新的模型
不引入时序
"""

def get_args():
	parser = argparse.ArgumentParser(description="APG Policy")
	
	parser.add_argument("--task", type=str, default="2DRotMov", help="The name of the task.")
	parser.add_argument("--experiment_name", type=str, default="Ver0", help="Name of the experiment to run or load.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed. Overrides config file if provided.")
	parser.add_argument("--device", type=str, default="cuda:0", help="The device")
	
	# train setting
	parser.add_argument("--learning_rate", type=float, default=1.6e-5, help="The learning rate of the optimizer")
	parser.add_argument("--batch_size", type=int, default=1024, help="Batch size of training. Notice that batch_size should be equal to num_envs")
	parser.add_argument("--num_worker", type=int, default=4, help="Number of workers for data loading")
	parser.add_argument("--num_epoch", type=int, default=400900, help="Number of epochs")
	parser.add_argument("--len_sample", type=int, default=5, help="Length of a sample")
	parser.add_argument("--slide_size", type=int, default=20, help="Size of GRU input window")
	
	# model setting
	parser.add_argument("--param_save_name", type=str, default='movingVer0.pth', help="The path to save model parameters")
	parser.add_argument("--param_load_name", type=str, default='movingVer0.pth', help="The path to load model parameters")
	
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

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # 如果使用 GPU，确保所有 CUDA 设备的随机性固定
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False  # 可能会降低某些情况下的性能，但保证了可复现性

if __name__ == "__main__":
	# 设置随机种子
	set_seed(42)
	# torch.autograd.set_detect_anomaly(True)
	args = get_args()
	run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{get_time()}"

	# 保存参数文件
	print(base_path)
	log_dir = os.path.join(base_path, 'runs/', run_name, "env.yaml")
	params = {
		k: v for k, v in args._get_kwargs()
	}
	save_dir = os.path.dirname(log_dir)
	os.makedirs(save_dir, exist_ok=True)
	dump_yaml(log_dir, params)

	writer_dir = os.path.join(base_path, 'runs/', run_name)
	writer = SummaryWriter(writer_dir)

	model_load_path = os.path.join(base_path, "param/", args.param_load_name)
	model_save_path = os.path.join(base_path, "param/", args.param_save_name)

	device = args.device
	print("using device:", device)


	envs = EnvMove(batch_size=args.batch_size, device=args.device)

	model = ZSLAModelVer2(image_dim=64, hidden_dim=256, query_num=10, num_classes=2, device=device)
	# model.load_model(path=model_load_path, device=device)

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	criterion_ce = nn.CrossEntropyLoss()
	criterion_mse = nn.MSELoss()

	
	for epoch in range(args.num_epoch):
		print(f"Epoch {epoch} begin...")
		optimizer.zero_grad()

		envs.reset(change_map=True)
		sum_loss = 0
		sum_loss_local_distance = 0
		sum_loss_local_class = 0
		sum_loss_global_exprate = 0
		sum_ave_loss = 0
		

		for step in range(args.len_sample):
			# print("step", step)
			step_output = envs.step()

			image = step_output["image"].detach()
			agent_pos = step_output["agent_pos_encode"].detach()
			gt = step_output["gt"]
			idx_reset = step_output["idx_reset"]

			output_local_distance, output_local_class, output_global_exprate = model(image, agent_pos, gt["local_query_encode"].detach(), gt["global_query_encode"].detach())

			# print("shape of output_local_distance", output_local_distance.shape)
			# print("shape of gt_local_distance", gt["local_gt_distance"].shape)
			loss_local_distance = criterion_mse(output_local_distance, gt["local_gt_distance"].detach())
			# print("shape of output_local_class", output_local_class.shape)
			# print("shape of gt_local_obstacle", gt["local_gt_obstacle"].shape)
			loss_local_class = criterion_ce(output_local_class.permute(0, 2, 1), gt["local_gt_obstacle"].detach())
			# print("shape of output_global_exprate", output_global_exprate.shape)
			# print("shape of gt_global_exprate", gt["global_explrate"].shape)
			loss_global_exprate = criterion_mse(output_global_exprate, gt["global_explrate"].detach())
			

			loss_total = loss_local_distance + loss_local_class + loss_global_exprate
			loss_total.backward()

			optimizer.step()
			optimizer.zero_grad()
			sum_loss += loss_total.detach()
			sum_loss_local_distance += loss_local_distance.detach()
			sum_loss_local_class += loss_local_class.detach()
			sum_loss_global_exprate += loss_global_exprate.detach()

			envs.reset()


		ave_sum_loss = sum_loss / args.len_sample

		writer.add_scalar('Loss', ave_sum_loss.item(), epoch)
		writer.add_scalar('Loss Local Distance', sum_loss_local_distance.item() / args.len_sample, epoch)
		writer.add_scalar('Loss Local Class', sum_loss_local_class.item() / args.len_sample, epoch)
		writer.add_scalar('Loss Global Exprate', sum_loss_global_exprate.item() / args.len_sample, epoch)

		if epoch % 2 == 0:
			print(f"Epoch {epoch} Loss: {ave_sum_loss.item()}")

		if not (epoch % 200) and epoch:
			print("Saving Model...")
			model.save_model(model_save_path)

	writer.close()
	print("Training Complete!")