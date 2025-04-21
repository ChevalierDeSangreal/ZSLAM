import torch
import torch.nn as nn

class ZSLAModel(nn.Module):
    """
    For env2d
    """
    def __init__(
        self,
        input_dim=66,      # 每帧输入的维度
        hidden_dim=64,     # GRU 的隐状态维度
        output_dim=3,      # 分类问题类别数：3 (对应 -1, 0, 1)
        device='cpu',
    ):
        super(ZSLAModel, self).__init__()
        
        self.hidden_dim = hidden_dim  # 方便在 forward 里使用
        self.device = device  # 保存设备信息

        # 双层 GRU，batch_first=True 方便处理 (batch, seq, feature)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        
        # Decoder: (h) -> 输出
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

        self.to(self.device)
    
    def forward(self, x, query, h=None):
        """
        单步前向传播:
          x: [batch_size, input_dim]  —— 单帧输入
          query: [batch_size, 3]      —— 额外输入信息
          h: [2, batch_size, hidden_dim] —— 上一时刻隐藏层 (可选)
        
        返回:
          out:    [batch_size, output_dim]
          new_h:  [2, batch_size, hidden_dim] —— 当前时刻输出的隐藏层，可用于下一步
        """
        # 如果没有传入上一时刻隐藏层，就初始化为0
        if h is None:
            h = torch.zeros(2, x.size(0), self.hidden_dim, device=x.device)
            # print("Here I am!!!!!!!!!!", x.device, h.device)
        
        # GRU 的输入需要 (batch_size, seq_len=1, input_dim)
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # out_seq: [batch_size, seq_len=1, hidden_dim]
        # new_h:   [2, batch_size, hidden_dim]
        out_seq, new_h = self.gru(x, h)
        
        # 因为 seq_len=1，所以我们把 out_seq squeeze 掉那一维
        out_seq = out_seq.squeeze(1)  # [batch_size, hidden_dim]
        
        # 拼接 (hidden_state) 和 query，并进入 Decoder
        out = self.decoder(torch.cat((out_seq, query), dim=1))  # [batch_size, output_dim]
        
        return out, new_h

    
    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cpu'):
        """加载模型参数"""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

def example_usage():
    # 假设我们有一个 batch_size=2，每个输入帧的维度=64
    batch_size = 2
    frame_dim = 64
    
    # 假设我们要连续处理 5 帧
    num_frames = 5
    
    # 随机生成模拟数据：形状 [num_frames, batch_size, frame_dim]
    # 代表5帧，每帧都有 batch_size=2 的样本
    fake_sequence = torch.randn(num_frames, batch_size, frame_dim)
    
    # 实例化模型
    model = ZSLAModel(input_dim=frame_dim, hidden_dim=64, output_dim=3)
    
    # 初始化隐藏层为 None
    h = None
    
    for t in range(num_frames):
        # 取第 t 帧数据: [batch_size, frame_dim]
        x_t = fake_sequence[t]  # shape: (2, 64)
        
        # 单步 forward
        out, h = model(x_t, h)
        
        # print(f"Frame {t} output shape: {out.shape}")
        # out.shape -> [2, 3]
        # h -> [1, batch_size=2, hidden_dim=64]  (留给下一帧使用)
    
    # 保存
    model.save_model("mynetwork_single_step.pth")

class ZSLAModelVer1(nn.Module):
    """
    For envmove
    Simplest one
    Use GRU as encoder
    Use MLP as decoder
    """
    def __init__(
        self,
        input_dim=80,      # 每帧输入的维度 深度相机像素数64+位置编码12+角度编码4
        hidden_dim=64,     # GRU 的隐状态维度
        output_dim=50,      # 真值相片总像素数
        num_classes=3,    # 分类问题类别数：3 (对应 0, 1, 2)
        device='cpu',
    ):
        super(ZSLAModelVer1, self).__init__()
        
        self.hidden_dim = hidden_dim  # 方便在 forward 里使用
        self.device = device  # 保存设备信息
        self.output_dim = output_dim
        self.num_classes = num_classes

        # 双层 GRU，batch_first=True 方便处理 (batch, seq, feature)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        
        # Decoder: (h) -> 输出
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 12, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim * num_classes),
        )

        self.to(self.device)
    
    def forward(self, x, query, h=None):
        """
        单步前向传播:
          x: [batch_size, input_dim]  —— 单帧输入
          query: [batch_size, 2]      —— 额外输入信息
          h: [2, batch_size, hidden_dim] —— 上一时刻隐藏层 (可选)
        
        返回:
          out:    [batch_size, output_dim]
          new_h:  [2, batch_size, hidden_dim] —— 当前时刻输出的隐藏层，可用于下一步
        """
        # 如果没有传入上一时刻隐藏层，就初始化为0
        if h is None:
            h = torch.zeros(2, x.size(0), self.hidden_dim, device=x.device)
            # print("Here I am!!!!!!!!!!", x.device, h.device)
        
        # GRU 的输入需要 (batch_size, seq_len=1, input_dim)
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # out_seq: [batch_size, seq_len=1, hidden_dim]
        # new_h:   [2, batch_size, hidden_dim]
        out_seq, new_h = self.gru(x, h)
        
        # 因为 seq_len=1，所以我们把 out_seq squeeze 掉那一维
        out_seq = out_seq.squeeze(1)  # [batch_size, hidden_dim]
        
        # 拼接 (hidden_state) 和 query，并进入 Decoder
        # print("out_seq.shape", out_seq.shape)
        # print("query.shape", query.shape)
        query_repeat = query.repeat(out_seq.shape[0], 1)
        out = self.decoder(torch.cat((out_seq, query_repeat), dim=1))  # [batch_size, output_dim]
        out = out.reshape(-1, self.output_dim, self.num_classes) # [batch_size, output_dim, num_classes]
        return out, new_h

    
    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cpu'):
        """加载模型参数"""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

def example_usage():
    # 假设我们有一个 batch_size=2，每个输入帧的维度=64
    batch_size = 2
    frame_dim = 64
    
    # 假设我们要连续处理 5 帧
    num_frames = 5
    
    # 随机生成模拟数据：形状 [num_frames, batch_size, frame_dim]
    # 代表5帧，每帧都有 batch_size=2 的样本
    fake_sequence = torch.randn(num_frames, batch_size, frame_dim)
    
    # 实例化模型
    model = ZSLAModel(input_dim=frame_dim, hidden_dim=64, output_dim=3)
    
    # 初始化隐藏层为 None
    h = None
    
    for t in range(num_frames):
        # 取第 t 帧数据: [batch_size, frame_dim]
        x_t = fake_sequence[t]  # shape: (2, 64)
        
        # 单步 forward
        out, h = model(x_t, h)
        
        print(f"Frame {t} output shape: {out.shape}")
        # out.shape -> [2, 3]
        # h -> [1, batch_size=2, hidden_dim=64]  (留给下一帧使用)
    
    # 保存
    model.save_model("mynetwork_single_step.pth")