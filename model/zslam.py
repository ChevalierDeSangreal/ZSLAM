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
        hidden_dim=256,     # GRU 的隐状态维度
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
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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


class ZSLAModelVer2(nn.Module):
    """
    For envmove
    Newly designed pretrain tasks
    MLP as image feature extractor
    MLP as encoder
    MLP as decoder
    """
    def __init__(
        self,
        image_dim=512,      # 每帧输入的维度 深度相机像素数
        hidden_dim=256,    # GRU 的隐状态维度
        query_num=10,      # 真值相片总像素数
        num_classes=2,     # 分类问题类别数：2 (对应 0, 1)
        device='cpu',
    ):
        super(ZSLAModelVer2, self).__init__()
        
        self.hidden_dim = hidden_dim  # 方便在 forward 里使用
        self.device = device  # 保存设备信息
        self.image_dim = image_dim

        self.query_num = query_num
        self.num_classes = num_classes

        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        # 双层 GRU，batch_first=True 方便处理 (batch, seq, feature)
        # self.local_encoder = nn.GRU(input_size=8+16, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        # self.global_encoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)

        self.local_encoder = nn.Sequential(
            nn.Linear(8 + 12 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.local_distance_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )
        self.local_class_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
        )

        self.global_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 12, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)
    
    def forward(self, image, agent_pos, local_query, global_query):
        """
        单步前向传播:
            image: [batch_size, 64]  当前时刻深度图
            agent_pos: [batch_size, 12]  当前时刻位姿编码
            local_query: [batch_size, num_query, 4]  局部查询
            global_query: [batch_size, 12]  全局查询
    
        返回:
            local_distance:    [batch_size, num_query, 1]  局部查询距离
            local_class:       [batch_size, num_query, num_classes]  局部查询分类
            global_exprate:    [batch_size, 1]  全局查询探索度
        """
        # 如果没有传入上一时刻隐藏层，就初始化为0
        # if h_local is None:
        #     h_local = torch.zeros(2, image.size(0), self.hidden_dim, device=image.device)
        # if h_global is None:
        #     h_global = torch.zeros(2, image.size(0), self.hidden_dim, device=image.device)

        image_feature = self.image_encoder(image)  # [batch_size, 8]

        intput_local_encoder = torch.cat((image_feature, agent_pos), dim=1)  # [batch_size, 8+12]
        
        # # GRU 的输入需要 (batch_size, seq_len=1, input_dim)
        # intput_local_encoder = intput_local_encoder.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # out_seq: [batch_size, seq_len=1, hidden_dim]
        # new_h:   [2, batch_size, hidden_dim]
        # out_seq, new_h_local = self.local_encoder(image, h_local)

        local_feature = self.local_encoder(intput_local_encoder)  # [batch_size, hidden_dim]

        # 扩展到 (batch_size, num_query, hidden_dim)
        # print("local_feature.shape", local_feature.shape)
        tmp_local_feature = local_feature.unsqueeze(1).expand(-1, local_query.size(1), -1)

        input_local_decoder = torch.cat((tmp_local_feature, local_query), dim=2)  # [batch_size, num_query, hidden_dim+4]
        # print("input_local_decoder.shape", input_local_decoder.shape)
        output_local_distance = self.local_distance_decoder(input_local_decoder)  # [batch_size, num_query, 1]
        output_local_class = self.local_class_decoder(input_local_decoder)  # [batch_size, num_query*num_classes]
        # output_local_class = output_local_class.reshape(-1, self.query_num, self.num_classes)  # [batch_size, num_query, num_classes]

        global_feature = self.global_encoder(local_feature)  # [batch_size, hidden_dim]
        input_global_decoder = torch.cat((global_feature, global_query), dim=1)  # [batch_size, hidden_dim+12]
        output_global_exprate = self.global_decoder(input_global_decoder)  # [batch_size, 1]
        
        
        return output_local_distance, output_local_class, output_global_exprate

    
    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cpu'):
        """加载模型参数"""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

class ZSLAModelVer3(nn.Module):
    """
    For envmove
    Newly designed pretrain tasks
    MLP as image feature extractor
    GRU as encoder
    MLP as decoder
    """
    def __init__(
        self,
        image_dim=512,      # 每帧输入的维度 深度相机像素数
        hidden_dim=256,    # GRU 的隐状态维度
        query_num=10,      # 真值相片总像素数
        num_classes=2,     # 分类问题类别数：2 (对应 0, 1)
        device='cpu',
    ):
        super(ZSLAModelVer3, self).__init__()
        
        self.hidden_dim = hidden_dim  # 方便在 forward 里使用
        self.device = device  # 保存设备信息
        self.image_dim = image_dim

        self.query_num = query_num
        self.num_classes = num_classes

        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        # 双层 GRU，batch_first=True 方便处理 (batch, seq, feature)
        self.local_encoder = nn.GRU(input_size=8 + 12 + 4, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.global_encoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)

        # self.local_encoder = nn.Sequential(
        #     nn.Linear(8 + 12 + 4, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, hidden_dim),
        # )

        # self.global_encoder = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.local_distance_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

        self.local_class_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
        )

        self.global_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 12, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)
    
    def forward(self, image, agent_pos, local_query, global_query, h_local=None, h_global=None):
        """
        单步前向传播:
            image: [batch_size, 64]  当前时刻深度图
            agent_pos: [batch_size, 12]  当前时刻位姿编码
            local_query: [batch_size, num_query, 4]  局部查询
            global_query: [batch_size, 12]  全局查询
            h_local: [2, batch_size, hidden_dim]  上一时刻局部隐藏层
            h_global: [2, batch_size, hidden_dim]  上一时刻全局隐藏层
    
        返回:
            local_distance:    [batch_size, num_query, 1]  局部查询距离
            local_class:       [batch_size, num_query, num_classes]  局部查询分类
            global_exprate:    [batch_size, 1]  全局查询探索度
            new_h_local:      [2, batch_size, hidden_dim]  当前时刻局部隐藏层
            new_h_global:     [2, batch_size, hidden_dim]  当前时刻全局隐藏层
        """
        # 如果没有传入上一时刻隐藏层，就初始化为0
        if h_local is None:
            h_local = torch.zeros(2, image.size(0), self.hidden_dim, device=image.device)
        if h_global is None:
            h_global = torch.zeros(2, image.size(0), self.hidden_dim, device=image.device)

        image_feature = self.image_encoder(image)  # [batch_size, 8]

        intput_local_encoder = torch.cat((image_feature, agent_pos), dim=1)  # [batch_size, 8+12]
        
        # GRU 的输入需要 (batch_size, seq_len=1, input_dim)
        intput_local_encoder = intput_local_encoder.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # out_seq: [batch_size, seq_len=1, hidden_dim]
        # new_h:   [2, batch_size, hidden_dim]
        local_feature_seq, new_h_local = self.local_encoder(intput_local_encoder, h_local)
        local_feature = local_feature_seq.squeeze(1)  # [batch_size, hidden_dim]

        # local_feature = self.local_encoder(intput_local_encoder)  # [batch_size, hidden_dim]

        # 扩展到 (batch_size, num_query, hidden_dim)
        # print("local_feature.shape", local_feature.shape)
        tmp_local_feature = local_feature.unsqueeze(1).expand(-1, local_query.size(1), -1)

        input_local_decoder = torch.cat((tmp_local_feature, local_query), dim=2)  # [batch_size, num_query, hidden_dim+4]
        # print("input_local_decoder.shape", input_local_decoder.shape)
        output_local_distance = self.local_distance_decoder(input_local_decoder)  # [batch_size, num_query, 1]
        output_local_class = self.local_class_decoder(input_local_decoder)  # [batch_size, num_query*num_classes]
        # output_local_class = output_local_class.reshape(-1, self.query_num, self.num_classes)  # [batch_size, num_query, num_classes]

        intput_local_feature = local_feature.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        global_feature_seq, new_h_global = self.global_encoder(intput_local_feature)
        global_feature = global_feature_seq.squeeze(1)  # [batch_size, hidden_dim]
        input_global_decoder = torch.cat((global_feature, global_query), dim=1)  # [batch_size, hidden_dim+12]
        output_global_exprate = self.global_decoder(input_global_decoder)  # [batch_size, 1]
        
        
        return output_local_distance, output_local_class, output_global_exprate, new_h_local, new_h_global

    
    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cpu'):
        """加载模型参数"""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

class ZSLAModelVer4(nn.Module):
    """
    Based on Ver3
    Added pure encoder mode
    Modified forward input to fit rl framework
    """
    def __init__(
        self,
        image_dim=512,      # 每帧输入的维度 深度相机像素数
        hidden_dim=256,    # GRU 的隐状态维度
        query_num=10,      # 真值相片总像素数
        num_classes=2,     # 分类问题类别数：2 (对应 0, 1)
        position_encode_size=12,  # 位置编码的维度
        attitude_encode_size=4,  # 姿态编码的维度
        device='cpu',
        is_pretrain=False,  # 是否为预训练模式
    ):
        # 建议直接修改默认参数，不要手动传参。因为在ppo网络里没有参数传入。除非手动修改encoder_ppo.py调用的相关部分
        super(ZSLAModelVer4, self).__init__()
        
        self.hidden_dim = hidden_dim  # 方便在 forward 里使用
        self.device = device  # 保存设备信息
        self.image_dim = image_dim

        self.query_num = query_num
        self.num_classes = num_classes
        self.is_pretrain = is_pretrain

        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        # 双层 GRU，batch_first=True 方便处理 (batch, seq, feature)
        self.local_encoder = nn.GRU(input_size=8 + 12 + 4, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.global_encoder = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)

        # self.local_encoder = nn.Sequential(
        #     nn.Linear(8 + 12 + 4, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, hidden_dim),
        # )

        # self.global_encoder = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )

        self.local_distance_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

        self.local_class_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
        )

        self.global_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 12, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)
    
    def forward(self, inputs):
        """
        单步前向传播:
            inputs: dict 包含以下键值：
                image: [batch_size, self.image_dim] 当前时刻深度图
                agent_pos: [batch_size, position_encode_size + attitude_encode_size] 当前时刻位姿编码
                h_local: [2, batch_size, hidden_dim] 上一时刻局部隐藏层（可选）
                h_global: [2, batch_size, hidden_dim] 上一时刻全局隐藏层（可选）

            训练阶段 (is_pretrain=True) 需要额外返回：
                local_distance:    [batch_size, num_query, 1] 局部查询距离
                local_class:       [batch_size, num_query, num_classes] 局部查询分类
                global_exprate:    [batch_size, 1] 全局探索度估计
            非预训练阶段 (is_pretrain=False) 返回：
                feature:           [batch_size, hidden_dim * 2] 联合特征表示
        """
        image = inputs['image']
        agent_pos = inputs['agent_pos_encode']
        h_local = inputs.get('h_local', None)
        h_global = inputs.get('h_global', None)

        if h_local is None:
            h_local = torch.zeros(2, image.size(0), self.hidden_dim, device=image.device)
        if h_global is None:
            h_global = torch.zeros(2, image.size(0), self.hidden_dim, device=image.device)

        image_feature = self.image_encoder(image)  # [batch_size, 8]
        input_local_encoder = torch.cat((image_feature, agent_pos), dim=1)  # [batch_size, 20]
        # print("????????????")
        # print("h_local.shape", h_local.shape)
        # print("h_global.shape", h_global.shape)
        # print("local_encoder", self.local_encoder)
        # print("input_local_encoder.shape", input_local_encoder.shape)
        input_local_encoder = input_local_encoder.unsqueeze(1)  # [batch_size, 1, 20]
        
        local_feature_seq, new_h_local = self.local_encoder(input_local_encoder, h_local)
        local_feature = local_feature_seq.squeeze(1)  # [batch_size, hidden_dim]

        input_local_feature = local_feature.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        global_feature_seq, new_h_global = self.global_encoder(input_local_feature, h_global)
        global_feature = global_feature_seq.squeeze(1)  # [batch_size, hidden_dim]

        if self.is_pretrain:
            # 以下字段在预训练阶段必须提供
            local_query = inputs['local_query']  # [batch_size, num_query, 4]
            global_query = inputs['global_query']  # [batch_size, 12]

            tmp_local_feature = local_feature.unsqueeze(1).expand(-1, local_query.size(1), -1)  # [batch_size, num_query, hidden_dim]
            input_local_decoder = torch.cat((tmp_local_feature, local_query), dim=2)  # [batch_size, num_query, hidden_dim + 4]

            output_local_distance = self.local_distance_decoder(input_local_decoder)  # [batch_size, num_query, 1]
            output_local_class = self.local_class_decoder(input_local_decoder)  # [batch_size, num_query, num_classes]

            input_global_decoder = torch.cat((global_feature, global_query), dim=1)  # [batch_size, hidden_dim + 12]
            output_global_exprate = self.global_decoder(input_global_decoder)  # [batch_size, 1]

            return output_local_distance, output_local_class, output_global_exprate, new_h_local, new_h_global

        else:

            feature = torch.cat((local_feature, global_feature), dim=1)  # [batch_size, hidden_dim * 2]
            return feature, new_h_local, new_h_global

        
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