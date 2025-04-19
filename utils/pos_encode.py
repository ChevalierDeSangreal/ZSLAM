import torch

def pos_encode(position, ori, device="cpu"):
    """
    位置编码
    :param position: 位置信息，形状为 (batch_size, 2)
    :param ori: 姿态信息，形状为 (batch_size,)
    :param device: 设备
    :return: 位置编码，形状为 (batch_size, 4)
    """
    ori = ori.unsqueeze(1)
    return torch.cat([position, torch.sin(ori), torch.cos(ori)], dim=1).to(device)

def position_encode(position, device="cpu"):
    """
    位置编码
    :param position: 位置信息，形状为 (batch_size, 2)
    :param device: 设备
    :return: 位置编码，形状为 (batch_size, 2)
    """
    return position.to(device)

def attitude_encode(ori, device="cpu"):
    """
    姿态编码
    :param ori: 姿态信息，形状为 (batch_size,)
    :param device: 设备
    :return: 姿态编码，形状为 (batch_size, 2)
    """
    return torch.stack([torch.sin(ori), torch.cos(ori)], dim=1).to(device)