import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import random

class CircleMap():
    def __init__(self, num_envs, device, map_r, num_component, max_r=2):

        # 初始化设置参数
        self.num_component = num_component
        self.num_envs = num_envs
        self.device = device
        self.map_r = map_r
        self.max_r = max_r

        self.output_path = '/home/wangzimo/VTT/ZSLAM/output'


        # 记录地图上所有物体的信息
        # xyr
        self.map_componet = torch.zeros((self.num_envs, num_component, 3)).to(self.device)
        
    
    def generate_map(self):
        """
        随机生成 num_component 个圆，每个圆满足：
          1. 圆心离原点的距离不超过 map_r
          2. 圆的边界（即圆心距离减半径）大于 0.5。
          
        实现思路：
          - 采用极坐标采样圆心，为保证在面积上均匀，先采样 d^2 在 [0.5^2, map_r^2] 内均匀分布，
            再令 d = sqrt(采样值)。
          - 圆心角度 theta 在 [0, 2π) 内均匀采样。
          - 对于每个圆，其允许的最大半径为 center_dist - 0.5
            在 (0, center_dist-0.5) 内均匀采样圆的半径
          - 假设圆所在平面 z 坐标固定为 0。
        """
        num_envs = self.num_envs
        num_component = self.num_component
        device = self.device

        # 定义圆心距离的下界和上界
        d_min = 0.5      # 为保证边界与原点距离 > 0.5，圆心距离必须大于 0.5
        d_max = self.map_r

        # 为保证在面积上均匀，先采样 d^2 在 [d_min^2, d_max^2] 内均匀分布
        rand_val = torch.rand((num_envs, num_component), device=device)
        d_squared = rand_val * (d_max**2 - d_min**2) + d_min**2
        center_dist = torch.sqrt(d_squared)  # 得到圆心离原点的距离

        # 采样圆心角度 theta, 范围 [0, 2π)
        theta = torch.rand((num_envs, num_component), device=device) * 2 * math.pi

        # 根据极坐标计算圆心 (x, y)
        x = center_dist * torch.cos(theta)
        y = center_dist * torch.sin(theta)

        # 对于每个圆，允许的最大半径为 (center_dist - 0.5)
        # 在 (0, center_dist-0.5) 内均匀采样圆的半径
        max_radius = center_dist - 0.5
        r_circle = torch.rand((num_envs, num_component), device=device) * max_radius


        # 更新 self.map_componet，顺序为 [x, y, r]
        self.map_componet = torch.stack([x, y, r_circle], dim=-1)


    def plot_map(self, env_idx=0):
        """
        可视化指定环境中的地图组件
        参数：
            env_idx (int): 环境索引，默认为0
        """
        
        components = self.map_componet[env_idx].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Environment {env_idx} Map")
        ax.set_xlim(-self.map_r - 1, self.map_r + 1)
        ax.set_ylim(-self.map_r - 1, self.map_r + 1)
        ax.set_aspect('equal')
        
        # 绘制安全区域（原点周围0.5半径）
        safe_circle = plt.Circle((0, 0), 0.5, color='gray', fill=False, linestyle='--', label='Safe Zone')
        ax.add_artist(safe_circle)
        
        # 绘制所有圆形组件
        for i, (x, y, r) in enumerate(components):
            circle = plt.Circle((x, y), r, color='blue', fill=False, label=f'Object {i+1}')
            ax.add_artist(circle)
        
        # 添加图例并显示
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:2], labels[:2])  # 仅显示前两个图例避免重复
        plt.grid(True)
        plt.savefig(self.output_path + '/map_visual.png')
        plt.show()

class CameraCoverage:
    def __init__(self, fov: float):
        """
        参数：
            fov (float): 相机视场角，单位为弧度
        """
        self.fov = fov
        # 对于 batch 中的每个相机，维护累计观测区间的左右边界（绝对弧度）
        self.observed_start = None  # shape: (batch_size,)
        self.observed_end = None    # shape: (batch_size,)

    def update(self, angles: torch.Tensor):
        """
        并行更新每个相机的累计观测区间。
        
        参数：
            angles (torch.Tensor): shape=(batch_size,)，表示当前相机角度（单位：弧度）
            
        说明：
            对于每个样本，新观测区间为 [angle - fov/2, angle + fov/2]
            假设旋转连续（边界单调），因此只需对右边界取最大值更新。
        """
        new_left = angles - self.fov / 2.0
        new_right = angles + self.fov / 2.0

        if self.observed_start is None:
            # 首次更新：记录初始区间
            self.observed_start = new_left.clone()
            self.observed_end = new_right.clone()
        else:
            # 连续旋转下，新区间的左边界一般满足 new_left <= observed_end，
            # 因此只需更新 observed_end
            self.observed_end = torch.max(self.observed_end, new_right)

    def reset(self):
        self.observed_start = None  # shape: (batch_size,)
        self.observed_end = None    # shape: (batch_size,)

    def query(self, query_angles: torch.Tensor) -> torch.Tensor:
        """
        输入 batch 个查询弧度 返回一个长度为 batch 的 tensor
        每个元素为 1 表示该角度已被观测到 0 表示未观测到。

        参数：
            query_angles (torch.Tensor): shape=(batch_size,)，各角度应处于 [0, 2π) 内（单位：弧度）。
        
        返回：
            torch.Tensor: shape=(batch_size,)，元素为 0 或 1。
        
        计算逻辑：
            1. 若累计覆盖角度 observed_end - observed_start >= 2π 则认为全覆盖 对应样本均返回 1。
            2. 否则先将累计区间映射到 [0,2π) 内：
                - 令 L = observed_start mod (2π), R = observed_end mod (2π).
            3. 若 L < R 则观测区间为 [L, R]，查询角度 q 在该区间内则认为已观测 1 否则未观测0。
            4. 若 L >= R 则说明累计区间跨越 0 即观测区间为 [L,2π) or [0,R] 
               查询角度满足 (q >= L) or (q <= R) 时为已观测。
        """
        if self.observed_start is None or self.observed_end is None:
            # 尚未更新任何状态，则均认为未观测
            return torch.zeros_like(query_angles, dtype=torch.int32)

        # 计算累计覆盖角度（弧度）
        coverage = self.observed_end - self.observed_start
        full_coverage_mask = coverage >= (2 * math.pi - 1e-6)  # shape: (batch_size,)

        # 将累计边界映射到 [0, 2π) 内
        L = torch.remainder(self.observed_start, 2 * math.pi)
        R = torch.remainder(self.observed_end, 2 * math.pi)

        # 对于每个样本：
        # 当 L < R 时，观测区间为 [L, R]，查询角度 q 满足 L <= q <= R 则返回 True
        # 当 L >= R 时，观测区间为 [L, 2π) ∪ [0, R]，查询角度 q 满足 (q >= L) or (q <= R) 则返回 True
        cond = L < R
        observed_case1 = (query_angles >= L) & (query_angles <= R)
        observed_case2 = (query_angles >= L) | (query_angles <= R)
        observed = torch.where(cond, observed_case1, observed_case2)

        # 对于累计覆盖 >= 2π 的样本，直接置为 True
        observed = torch.where(full_coverage_mask, torch.ones_like(observed, dtype=torch.bool), observed)

        return observed.int()

    

class TwoDEnv():
    def __init__(self, num_envs, device):
        
        # 初始化设置参数
        self.num_envs = num_envs
        self.device = device
        self.fov = 2 * math.pi * 60 / 360
        self.dt = 0.02
        self.num_rays = 64

        self.max_ang_vel = math.radians(60)  # 60° 转换为弧度

        self.output_path = '/home/wangzimo/VTT/ZSLAM/output'

        # 定义无人机状态
        # (朝向，角速度，角加速度)
        self.quad_state = torch.zeros((self.num_envs, 3)).to(self.device)
        self.observed = CameraCoverage(self.fov)
        self.last_step = torch.zeros((self.num_envs,)).to(self.device) # 每个step-1，归零时使用新的角加速度或相反的角加速度，由phase switch决定

        # 当前加速阶段标识（True 表示加速阶段，False 表示减速阶段）
        self.is_accelerating = True
        self.phase_steps = random.randint(10, 50)

        # 定义地图组件
        self.map = CircleMap(num_envs=num_envs, device=device, map_r=5, num_component=10)


    def reset(self):
        self.observed.reset()
        self.quad_state.zero_()
        self.quad_state[:, 0] = torch.rand((self.num_envs,), device=self.device) * 2 * math.pi  # 随机初始化初始朝向
        self.last_step.zero_()  # 将所有剩余步数置 0，以便下一步重置时重新生成新的加速度

        self.is_accelerating = True
        self.phase_steps = random.randint(10, 50)

        self.map.generate_map()

    def step(self):
        """更新无人机状态，并控制角速度变化，使其在加速与减速阶段循环变化"""
        # 检查剩余步数是否归零：如果归零则需要重新生成角加速度（对于整个 batch 都相同）
        if self.last_step[0].item() == 0:
            if self.is_accelerating:
                # 随机生成本阶段的持续步数（例如 10 到 50 步之间）
                self.phase_steps = random.randint(10, 50)
                # 根据最大角速度和持续时间计算角加速度大小
                # 注意：phase_steps * dt 为本阶段的持续时间
                current_acc = self.max_ang_vel / (self.phase_steps * self.dt)
                self.quad_state[:, 2].fill_(current_acc)
                # 重置所有环境的剩余步数
                self.last_step.fill_(self.phase_steps)
                
            else:
                self.quad_state[:, 2] *= -1
                self.last_step.fill_(self.phase_steps)

            self.is_accelerating = not self.is_accelerating

        # 更新角速度： ω = ω + α * dt
        self.quad_state[:, 1] += self.quad_state[:, 2] * self.dt
        # self.quad_state[:, 1] = torch.clamp(self.quad_state[:, 1], -self.max_ang_vel, self.max_ang_vel)
        # 更新朝向： θ = θ + ω * dt
        self.quad_state[:, 0] += self.quad_state[:, 1] * self.dt

        # 减少剩余步数（所有环境保持一致）
        self.last_step -= 1

        # 更新观测状态（假设 CameraCoverage 提供 update 方法，根据 quad_state 更新观测信息）
        self.observed.update(self.quad_state[:, 0])

        # 生成训练数据，返回表示方式为 [cos(theta), sin(theta), distance]
        training_points = self.generate_training_points()

        # 获取当前无人机朝向
        current_angle = self.quad_state[:, 0]
        # 渲染深度相机数据
        depth_obs = self.render(current_angle, self.fov, num_rays=self.num_rays)
        # 根据训练点生成地面真值标签
        gt_labels = self.generate_ground_truth(training_points)

        # 对当前无人机朝向进行二元编码表示
        quad_angle_enc = self.angle_encoding(current_angle)

        return depth_obs.detach(), quad_angle_enc.detach(), training_points.detach(), gt_labels.detach()
    
    def generate_training_points(self):
        """
        生成每个环境一个随机点，范围在地图半径内。
        返回形式为 [cos(theta), sin(theta), distance]，其中 theta 为点的极角，distance 为到原点的距离。
        """
        num_envs = self.num_envs
        map_r = self.map.map_r  # 获取地图半径
        
        # 使用极坐标采样保证均匀分布
        d_squared = torch.rand((num_envs, 1), device=self.device) * (map_r**2)
        r = torch.sqrt(d_squared)  # 距离
        theta = torch.rand((num_envs, 1), device=self.device) * 2 * math.pi  # 极角
        
        # 使用 angle_encoding 方法得到二元编码（cos, sin）
        encoding = self.angle_encoding(theta.squeeze(-1))  # shape: (num_envs, 2)
        # 拼接距离信息，得到 (num_envs, 3)
        training_points = torch.cat([encoding, r], dim=1)
        return training_points

    def generate_ground_truth(self, points):
        """
        生成地面真值标签
        :param points: 输入点坐标 (num_envs, 2)
        :return: 标签张量 (num_envs,), 0=未观测, 1=未遮挡, -1=被遮挡
        """
        # 恢复点的 (x, y) 坐标
        x = points[:, 0] * points[:, 2]
        y = points[:, 1] * points[:, 2]
        
        # 计算目标点的方位角，并标准化到[0, 2π)
        theta_points = torch.atan2(y, x)
        theta_points = torch.remainder(theta_points, 2 * math.pi)
        
        # 为每个目标点生成独立的渲染结果
        # 并行处理所有点的渲染请求
        cam_angles = theta_points  # 使用目标点的方向作为相机朝向
        depth_obs = self.render(cam_angles, self.fov, num_rays=self.num_rays)[:, self.num_rays // 2]
        
        # 掩码：将未观测到的点标记为0
        observed = self.observed.query(theta_points)
        gt = torch.zeros_like(observed, dtype=torch.float32)
        
        # 仅处理已观测点
        mask = observed.bool()
        if not mask.any():
            return gt
        
        # 计算遮挡状态
        distances = torch.sqrt(x**2 + y**2)
        occluded = distances > depth_obs
        
        # 更新标签
        gt_mask = mask
        gt[gt_mask] = torch.where(occluded[gt_mask], -1, 1).to(gt.dtype)
        
        return gt
    
    def render(self, cam_angle, fov, num_rays=64, far_clip=4.0):
        """
        利用深度相机原理对圆形物体进行渲染，返回每个环境中各射线方向上距离最近交点的深度值。
        假设原点始终处于圆的外部。

        参数：
            cam_angle: 相机朝向（弧度）
            fov: 相机视野角度（弧度）
            num_rays: 射线数量（在 fov 内均匀采样）
            far_clip: 当射线未击中任何圆时赋予的远裁剪距离 默认 4.0

        返回：
            depth: 形状 (num_envs, num_rays) 的张量，每个元素表示对应射线的深度值
        """
        # 1. 生成射线角度（形状：(num_rays,)），并扩展维度以便广播
        ray_angles = cam_angle.unsqueeze(-1) + torch.linspace(-fov/2, fov/2, steps=num_rays, device=self.device)
        ray_angles = ray_angles.view(self.num_envs, 1, num_rays)  # 形状：(1, 1, num_rays)

        # 2. 提取每个圆的中心 (x, y) 和半径 r
        centers = self.map.map_componet[:, :, :2]  # 形状：(num_envs, num_component, 2)
        radii = self.map.map_componet[:, :, 2:3]   # 形状：(num_envs, num_component, 1)

        # 3. 计算原点到圆心的距离 d 及其角度 φ
        d = torch.norm(centers, dim=2, keepdim=True)            # 形状：(num_envs, num_component, 1)
        phi = torch.atan2(centers[..., 1:2], centers[..., 0:1]) # 形状：(num_envs, num_component, 1)

        # 4. 计算射线与圆心之间的相对角度 δ，并归一化到 [-π, π]
        delta = ray_angles - phi                                # 形状：(num_envs, num_component, num_rays)
        delta = (delta + math.pi) % (2 * math.pi) - math.pi

        # 5. 计算射线与圆的交点：
        D = radii**2 - (d * torch.sin(delta))**2               # 形状：(num_envs, num_component, num_rays)
        sqrt_D = torch.sqrt(torch.clamp(D, min=0.0))
        L = d * torch.cos(delta)                                # 计算 d * cos(δ)
        t_candidate = L - sqrt_D                                # 交点距离

        # 6. 判定有效交点：要求 D >= 0 且 t_candidate > 0
        valid_mask = (D >= 0) & (t_candidate > 0)
        t_candidate_valid = torch.where(valid_mask, t_candidate, far_clip * torch.ones_like(t_candidate))

        # 7. 选取每个射线的最近交点作为最终深度
        depth, _ = torch.min(t_candidate_valid, dim=1)          # 形状：(num_envs, num_rays)

        return depth

    def visualize_render(self, depth, cam_angle, fov, env_idx=0):
        """
        可视化渲染的深度结果
        参数：
            depth (torch.Tensor): render方法返回的深度张量
            cam_angle (float): 相机朝向（弧度）
            fov (float): 相机视场角（弧度）
            env_idx (int): 环境索引 默认为0
        """
        
        # 获取当前环境的地图
        self.map.plot_map(env_idx)
        ax = plt.gca()
        
        # 提取深度数据（仅第一个环境）
        rays = depth[env_idx].cpu().numpy()
        num_rays = len(rays)
        
        # 生成射线角度
        angles = torch.linspace(-fov/2, fov/2, num_rays) + cam_angle
        angles = angles.cpu().numpy()
        
        # 计算射线终点坐标
        x = rays * np.cos(angles)
        y = rays * np.sin(angles)
        
        # 绘制射线终点
        ax.scatter(x, y, color='red', s=10, label='Ray Hits')
        
        # 绘制相机视角锥形区域
        start_angle = cam_angle - fov/2
        end_angle = cam_angle + fov/2
        for angle in [start_angle, end_angle]:
            dx = np.cos(angle)
            dy = np.sin(angle)
            ax.plot([0, 10*dx], [0, 10*dy], 'g--', alpha=0.5)
        
        plt.legend()
        plt.savefig(self.output_path + '/render_visual.png')
        plt.show()

    @staticmethod
    def angle_encoding(angle: torch.Tensor) -> torch.Tensor:
        """
        输入角度，返回由余弦和正弦组成的二元编码。
        参数：
            angle (torch.Tensor): 形状 (...,)，角度（弧度）
        返回：
            torch.Tensor: 对应的二元编码，形状 (..., 2)
        """
        return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)