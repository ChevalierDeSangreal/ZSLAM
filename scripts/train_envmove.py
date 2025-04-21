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

def get_args():
	parser = argparse.ArgumentParser(description="APG Policy")
	
	parser.add_argument("--task", type=str, default="2DRotMov", help="The name of the task.")
	parser.add_argument("--experiment_name", type=str, default="Ver0", help="Name of the experiment to run or load.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed. Overrides config file if provided.")
	parser.add_argument("--device", type=str, default="cuda:0", help="The device")
	
	# train setting
	parser.add_argument("--learning_rate", type=float, default=5.6e-6, help="The learning rate of the optimizer")
	parser.add_argument("--batch_size", type=int, default=512, help="Batch size of training. Notice that batch_size should be equal to num_envs")
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
	torch.autograd.set_detect_anomaly(True)
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

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	envs = EnvMove(batch_size=args.batch_size, device=args.device)

	model = ZSLAModelVer1(input_dim=80, hidden_dim=64, output_dim=2500, device=device)
	# model.load_model(path=model_load_path, device=device)

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-6)
	criterion = nn.CrossEntropyLoss(reduction='none')  # 每个像素独立损失

	no_reset_buf = torch.ones(args.batch_size, device=device)
	
	for epoch in range(args.num_epoch):
		print(f"Epoch {epoch} begin...")
		optimizer.zero_grad()

		h0 = None
		envs.reset()
		sum_loss = torch.zeros(args.batch_size, device=device)
		sum_ave_loss = 0
		

		for step in range(args.len_sample):
			# print("step", step)
			step_output = envs.step()

			# print(step_output["image"].shape, step_output["agent_pos_encode"].shape, step_output["gt_position_encode"].shape)
			# print(step_output["image"], step_output["agent_pos_encode"])
			input_tmp = torch.cat((step_output["image"], step_output["agent_pos_encode"]), dim=1)
			if torch.isnan(input_tmp).any():
				print("NaN detected in intput")
			if torch.isnan(step_output["gt_position_encode"]).any():
				print("NaN detected in gt_position_encode")
			# print("gt_position_encode", step_output["gt_position_encode"])
			output, h0 = model(input_tmp, step_output["gt_position_encode"], h0)
			# print(output[0], gt_labels[0])
			# print("output shape", output.shape)
			# print("gt shape", step_output["gt"].shape)
			# print(type(output), type(step_output["gt"]))
			target = step_output["gt"].long()
			loss = criterion(output.permute(0,2,1), target).mean(dim=1) # [batch_size]
			# loss = criterion(output.float(), step_output["gt"].float()).mean(dim=1)
			if torch.isnan(output).any():
				print(input_tmp)
				print("NaN detected in output")
				exit(0)

			if torch.isnan(step_output["gt"]).any():
				print("NaN detected in ground truth (gt)")

			h0 = h0.clone()
			h0[:, step_output["idx_reset"]] = 0
			# print("loss shape", loss.shape)
			# print("sum_loss shape", sum_loss.shape)
			sum_loss += loss

			if (not (step + 1) % 50):
			
				# print(type(no_reset_buf))
				# no_reset_buf[step_output["idx_reset"]] = 1
				sum_loss.backward(no_reset_buf)
				# no_reset_buf *= 0
				
				optimizer.step()
				optimizer.zero_grad()
				h0 = h0.detach()

				sum_ave_loss += sum_loss.mean()
				sum_loss = torch.zeros(args.batch_size, device=device)

				envs.reset()

		print("Sum Ave Loss", sum_ave_loss)
		writer.add_scalar('Loss', sum_ave_loss.item(), epoch)

		if not (epoch % 200) and epoch:
			print("Saving Model...")
			model.save_model(model_save_path)

	writer.close()
	print("Training Complete!")