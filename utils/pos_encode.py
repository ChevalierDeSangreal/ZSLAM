
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.kinematics import batch_theta_to_rotation_matrix


def pos_encode(position, ori, theta=0.1, device="cpu"):
    """
    位置编码
    :param position: 位置信息，形状为 (batch_size, 2)
    :param ori: 姿态信息，形状为 (batch_size,)
    :param device: 设备
    :return: 位置编码，形状为 (batch_size, 16)
    """
    assert position.shape[0] == ori.shape[0]

    p, a = position_encode(position, theta, device=device), attitude_encode(ori, device=device)
    return torch.cat([p, a], dim=1)

def position_encode(position, theta=0.1, device="cpu"):
    """
    位置编码
    :param position: 位置信息，形状为 (batch_size, 2)
    :param device: 设备
    :return: 位置编码，形状为 (batch_size, 12)
    """
    B, _ = position.shape
    ones = torch.ones(B, 1).to(device)
    R = batch_theta_to_rotation_matrix(theta * torch.cat([position.to(device), ones], dim=1)).reshape(-1, 12) + 0.
    return R

def attitude_encode(ori, device="cpu"):
    """
    姿态编码
    :param ori: 姿态信息，形状为 (batch_size,)
    :param device: 设备
    :return: 姿态编码，形状为 (batch_size, 4)
    """
    att_encode = torch.stack([torch.cos(ori), -torch.sin(ori), torch.sin(ori), torch.cos(ori)], dim=1).to(device) + 0.
    return att_encode

if __name__ == '__main__':
    ori = torch.tensor([3.141592653589793, 0, 3.141592653589793*0.5])
    print(attitude_encode(ori).shape)
    position = torch.tensor([[1., 1.], [20., 1.], [5.25, -1.25]], dtype=torch.float32)
    print(position_encode(position).shape)
    poe = pos_encode(position, ori, device="cpu")
    print(poe.shape)
