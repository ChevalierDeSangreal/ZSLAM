import torch
from map import Map
from utils.plot import bool_tensor_visualization
from utils.geometry import get_circle_points, get_line_points, transformation, transformation_back, trans_simple, trans_simple_back
from utils import batch_theta_to_rotation_matrix, batch_theta_to_orientation_vector
from utils import position_encode, pos_encode, attitude_encode
from typing import List
import math
from cfg import *

class Agent:
    def __init__(self, agent_cfg:AgentCfg, batch_size, dt=0.02, ori=None, device='cpu'):
        """
        初始化无人机代理 配置相机参数 配置无人机参数 配置无人机位置和朝向

        参数:
            agent_cfg (CameraCfg): 相机配置参数，包括焦距、视场角、图像宽度等。
            ori (float): 相机朝向角度（弧度制），若未提供，则随机初始化。
            batch_size (int): 批大小，即同时模拟的相机数量。
            dt (float): 时间步长，默认为 0.02。
            device (str): 计算设备，默认为 'cpu'。

        主要属性:
            f (torch.tensor): 相机焦距。
            field (torch.tensor): 视场角（弧度制），要求在 (0, π) 范围内。
            w (int): 图像的像素宽度。
            field_radius (torch.tensor): 视场半径。
            ori (torch.tensor): 相机朝向角度（弧度制），若未提供，则随机初始化。
            R (torch.tensor): 旋转矩阵，由朝向角度计算得到。
            ori_vector (torch.tensor): 方向向量，由朝向角度计算得到。
            safe_radius (torch.tensor): 安全半径，若未提供，则计算 `f / sin(0.5 * field)`。

        说明:
            - 若 `agent_cfg.ori` 未提供，则 `ori` 采用 [0, 2π) 范围内的随机值初始化。
            - 旋转矩阵 `R` 和方向向量 `ori_vector` 由 `ori` 计算得到，用于表示相机的方向信息。
            - `safe_radius` 是基于焦距 `f` 和视场角 `field` 计算的最小安全距离。
        """
        assert 0 < agent_cfg.field < math.pi, 'Error::in Agent __init__: Wrong field angle'
        self.cfg = agent_cfg

        self.batch_size = batch_size
        self.device = device

        self.dt = torch.tensor(dt, dtype=torch.float, device=device)
        
        # 将所有浮点数转换为 tensor.float 类型
        self.f = torch.tensor(agent_cfg.f, dtype=torch.float, device=device)
        self.field = torch.tensor(agent_cfg.field, dtype=torch.float, device=device)
        self.w = agent_cfg.w
        self.field_radius = torch.tensor(agent_cfg.field_radius, dtype=torch.float, device=device)
        
        self.pos = torch.zeros((batch_size, 2), dtype=torch.float, device=device)
        # 随机初始化ori（若未提供）
        self.ori = torch.tensor(ori, dtype=torch.float, device=device).expand(batch_size) if ori is not None else torch.rand((batch_size, ), dtype=torch.float, device=device) * 2 * math.pi
        
        # print("Shape of ori:", self.ori.shape)

        # 计算旋转矩阵和方向向量
        self.R = batch_theta_to_rotation_matrix(self.ori)
        self.ori_vector = batch_theta_to_orientation_vector(self.ori)
        # print("Shape of ori_vector:", self.ori_vector.shape)
        # 计算安全半径
        self.safe_radius = torch.tensor(agent_cfg.safe_radius, dtype=torch.float, device=device) if agent_cfg.safe_radius is not None else self.f / torch.sin(0.5 * self.field)

        self.vel = torch.zeros((batch_size, 2), dtype=torch.float, device=device)
        self.acc = torch.zeros((batch_size, 2), dtype=torch.float, device=device)
        self.prefer_acc = torch.rand((batch_size, 2), dtype=torch.float, device=device)
        self.att_vel = torch.zeros((batch_size,), dtype=torch.float, device=device)
        self.att_acc = torch.zeros((batch_size,), dtype=torch.float, device=device)

        self.att_acc_timer = torch.zeros((batch_size,), dtype=torch.int, device=device) # 计时器，记录角加速度未变化的时间
        self.att_acc_change_time = torch.randint(agent_cfg.min_att_acc_change_step, agent_cfg.max_att_acc_change_step, (batch_size,), device=device) # 角加速度变化时间间隔

        self.desired_pos = torch.zeros((batch_size, 2), dtype=torch.float, device=device)

    def step(self):
        # 根据当前加速度和角加速度更新速度、角速度、位置和朝向
        self.pos = self.pos + 0.5 * self.vel * self.dt + 0.5 * self.acc * self.dt ** 2
        # print(self.pos[0])
        self.vel = self.vel + self.dt * self.acc
        self.vel = torch.clamp(self.vel, -self.cfg.max_speed, self.cfg.max_speed)
        # print("Before ori:", self.ori.shape)
        # print(self.ori.shape, self)
        self.ori = self.ori + 0.5 * self.att_vel * self.dt + 0.5 * self.att_acc * self.dt ** 2
        # print("After ori:", self.ori.shape)
        self.att_vel = self.att_vel + self.dt * self.att_acc
        self.att_vel = torch.clamp(self.att_vel, -self.cfg.max_att_speed, self.cfg.max_att_speed)

        self.R = batch_theta_to_rotation_matrix(self.ori)
        # print("Before ori_vector:", self.ori_vector.shape)
        self.ori_vector = batch_theta_to_orientation_vector(self.ori)
        # print("After ori_vector:", self.ori_vector.shape)

        self.att_acc_timer += 1

    def reset_idx(self, idx, init_pos, desired_pos):
        num_reset = len(idx)
        # 随机初始化无人机初始位置和姿态
        self.ori[idx] = torch.rand((num_reset, ), dtype=torch.float, device=self.device) * 2 * math.pi
        self.R[idx] = batch_theta_to_rotation_matrix(self.ori[idx])
        self.ori_vector[idx] = batch_theta_to_orientation_vector(self.ori[idx])
        # print("Shape of ori_vector:", self.ori_vector.shape)
        self.pos[idx] = init_pos[idx]
        self.vel[idx] = 0
        self.att_vel[idx] = 0

        self.prefer_acc = torch.rand((num_reset, 2), dtype=torch.float, device=self.device) * 2 - 1
        self.att_acc_timer[idx] = 0
        self.att_acc_change_time[idx] = torch.randint(self.cfg.min_att_acc_change_step, self.cfg.max_att_acc_change_step, (num_reset,), device=self.device)

        self.desired_pos[idx] = desired_pos[idx]
    

        

class EnvMove:
    def __init__(
            self, 
            batch_size:int,
            resolution_ratio=0.0,
            device="cpu",
            ):
        # 初始化环境之后得手动调用一次reset，用于生成智能体状态、更新可视范围
        # resolution_ratio < 0 不渲染
        """
            w_gt (int): ground truth 采样像素宽度，应该注意，w的取值应大于等于 `tan(alpha/2) / tan(ratio/2R) + 1` 建议优化为 `2R*tan(alpha/2) / ratio + 1`
                        之所以需要 w_gt 和 w 同时存在，是因为w_gt算出来可能太大了（分辨率0.01，视场角为90°时，w会逼近1000），而现实中分辨率不高的场景很普遍
                        由于ratio是env的属性，所以w_gt应在EnvMove的init_grid方法中实现。
        """
        self.batch_size = batch_size
        self.device = device

        self.cfg = EnvMoveCfg()

        self.agent = Agent(self.cfg.agent_cfg, batch_size, dt=self.cfg.dt, device=device)

        self.map = Map(self.cfg.map_cfg)
        self.map.random_initialize()

        self.__resolution_ratio = resolution_ratio


        self.init_grid()
        if resolution_ratio > 0:
            H, W = int(self.map.height/resolution_ratio), int(self.map.width/resolution_ratio)
            self.H, self.W = H, W
            self.grid = torch.zeros((H, W), device=device, dtype=torch.bool)
            self.center = torch.tensor([H//2, W//2], dtype=torch.int32, device=device)
            self.init_visual_grid()
            bool_tensor_visualization(self.grid.to("cpu"))


        

    def init_grid(self):
        ratio = self.map.ratio
        self.__ratio = ratio
        
        self.H, self.W = int(self.map.height/ratio), int(self.map.width/ratio)

        w_gt =  (2 * self.agent.field_radius * torch.tan(0.5 * self.agent.field)) / ratio + 1
        w_gt = torch.ceil(w_gt).to(torch.int32).item()
        w_gt = min(w_gt, self.H + self.W)
        self.w_gt = w_gt
        
        # n_theta = int(2 * math.pi * self.agent.field_radius / ratio)
        # n_theta = min(n_theta, 2 * (self.W + self.H))
        # self.n_theta = n_theta
        # print(self.n_theta)

        self.map_center = torch.tensor([self.W//2, self.H//2], dtype=torch.float32, device=self.device)
        if len(self.map.circle_center_array) != 0:
            self.circle_center = torch.stack(self.map.circle_center_array).to(self.device)
            self.circle_radius = torch.stack(self.map.circle_radius_array).to(self.device)
        if len(self.map.line_array) != 0:
            self.line = torch.stack(self.map.line_array).to(self.device)
        if len(self.map.triangle_point_array) != 0:
            self.triangle_points = torch.stack(self.map.triangle_point_array).to(self.device)
        points_all, points_no_obstacle, points_obs, point_mask_obstacle, grid, grid_mask_obstacle = self.get_map_grid(self.map)

        # 这里需要用transformation_back是因为从(0, 0)(W, H)平移到(-W/2, -H/2)(W/2, H/2)的坐标系
        # 下面这几个都是[batch_size, number of points, 2]的点集，其中2是float形式的物理xy坐标
        self.points_all = trans_simple_back(points_all, self.map_center) * ratio # 所有点的xy坐标
        self.points_no_obstacle = trans_simple_back(points_no_obstacle, self.map_center) * ratio # 无障碍物的点的xy坐标
        self.points_obstacle = trans_simple_back(points_obs, self.map_center) * ratio # 障碍物的点的xy坐标
        self.points_safe, self.safe_grid_mask_obstacle = self.get_safe_set()
        self.points_safe = trans_simple_back(self.points_safe, self.map_center) * ratio # 安全集合的点的xy坐标

        # 下面这三个都是[W, H, 2]的矩阵。实际物理坐标/ratio取整再平移到(-W/2, -H/2)(W/2, H/2)的坐标系就对应矩阵下标
        self.grid_map = grid # 存的就是实际物理坐标了
        # print("Shape of grid:", self.grid_map.shape)
        self.grid_mask_obstacle = grid_mask_obstacle
        self.grid_safe_mask_obstacle = self.safe_grid_mask_obstacle

        self.points_visible = torch.zeros(self.batch_size, self.W*self.H, dtype=torch.bool, device=self.device)
        self.grid_mask_visible = torch.zeros(self.batch_size, self.W, self.H, dtype=torch.bool, device=self.device)
        self.grid_visit_time = torch.zeros(size=(self.batch_size, self.W, self.H), dtype=torch.float32, device=self.device)
        
        

    def physical_to_matrix(self, physical_coords):
        """
        将物理坐标转换为矩阵坐标，并进行四舍五入。
        physical_coords: Tensor [N, 2]，其中每行是 (x, y)
        返回矩阵坐标 Tensor [N, 2]，其中每行是 (row, col)
        """
        x, y = physical_coords[:, 0], physical_coords[:, 1]
        row = torch.round(self.H // 2 + y / self.__ratio).long()
        col = torch.round(x / self.__ratio + self.W // 2).long()
        return torch.stack((col, row), dim=-1)

    def init_visual_grid(self):
        points = None

        if len(self.map.circle_center_array) != 0:
            circle_center = self.circle_center / self.__resolution_ratio
            circle_radius = self.circle_radius / self.__resolution_ratio
            points = get_circle_points(circle_center, circle_radius)

        if len(self.map.line_array) != 0:
            line = self.line / self.__resolution_ratio
            line_points = get_line_points(lines=line)
            if points is not None:
                points = torch.cat([points, line_points], dim=0)
            else:
                points = line_points
        
        if points is not None:
            points = transformation(points, self.center)
            valid = (points[:, 0] >= 0) & (points[:, 0] < self.H) & (points[:, 1] >= 0) & (points[:, 1] < self.W)
            points = points[valid]
            self.grid[points[:, 0], points[:, 1]] = True

    def get_safe_set(self):
        r = torch.max(self.agent.safe_radius).item()
        safe_map = Map(
            self.cfg.map_cfg
        )
        for i in range(len(self.map.circle_center_array)):
            safe_map.add_circle(
                x=self.map.circle_center_array[i][0].item(),
                y=self.map.circle_center_array[i][1].item(),
                r=self.map.circle_radius_array[i][0].item()+r
            )
        for i in range(len(self.map.triangle_point_array)):
            p0, p1, p2 = self.map.triangle_point_array[i][0], self.map.triangle_point_array[i][1], self.map.triangle_point_array[i][2]
            v0, v1, v2 = p1-p2, p2-p0, p0-p1
            v0_, v1_, v2_ = torch.tensor([v0[1], -v0[0]]), torch.tensor([v1[1], -v1[0]]), torch.tensor([v2[1], -v2[0]])
            v0_, v1_, v2_ = v0_/torch.norm(v0_)*r, v1_/torch.norm(v1_)*r, v2_/torch.norm(v2_)*r
            safe_map.add_triangle(p1.tolist(), (p1+v0_).tolist(), (p2+v0_).tolist())
            safe_map.add_triangle(p1.tolist(), (p1-v0_).tolist(), (p2-v0_).tolist())
            safe_map.add_triangle(p2.tolist(), p1.tolist(), (p2+v0_).tolist())
            safe_map.add_triangle(p2.tolist(), p1.tolist(), (p2-v0_).tolist())

            safe_map.add_triangle(p2.tolist(), (p2+v1_).tolist(), (p0+v1_).tolist())
            safe_map.add_triangle(p2.tolist(), (p2-v1_).tolist(), (p0-v1_).tolist())
            safe_map.add_triangle(p0.tolist(), p2.tolist(), (p0+v1_).tolist())
            safe_map.add_triangle(p0.tolist(), p2.tolist(), (p0-v1_).tolist())

            safe_map.add_triangle(p0.tolist(), (p0+v2_).tolist(), (p1+v2_).tolist())
            safe_map.add_triangle(p0.tolist(), (p0-v2_).tolist(), (p1-v2_).tolist())
            safe_map.add_triangle(p1.tolist(), p0.tolist(), (p1+v2_).tolist())
            safe_map.add_triangle(p1.tolist(), p0.tolist(), (p1-v2_).tolist())

            safe_map.add_triangle(p0.tolist(), p1.tolist(), p2.tolist())
            safe_map.add_circle(x=p0[0].item(), y=p0[1].item(), r=r)
            safe_map.add_circle(x=p1[0].item(), y=p1[1].item(), r=r)
            safe_map.add_circle(x=p2[0].item(), y=p2[1].item(), r=r)
        #dists = torch.cdist(safe, block)
        _, safe_points, _, _, _, safe_grid_mask = self.get_map_grid(safe_map)
        return safe_points, safe_grid_mask

    def is_collision(self):
        # origins = torch.stack([camera.position.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        # r = torch.tensor([camera.safe_radius for camera in self.cameras], device=self.device).unsqueeze(-1).unsqueeze(-1)
        origins = self.agent.pos.unsqueeze(1)
        # print("??????????????", origins.shape)
        r = self.agent.safe_radius.unsqueeze(-1).unsqueeze(-1)
        # r = 0
        is_collision = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        if len(self.map.circle_center_array) != 0:
            distance = torch.norm(self.circle_center-origins, dim=-1, keepdim=True)
            sign = (distance - r - self.circle_radius <=  0)
            is_collision |= sign.squeeze(-1).any(dim=-1, keepdim=False)

        if len(self.map.line_array) != 0:
            line = self.line.unsqueeze(0)   # 1, n, 2, 2
            d = line[..., 1, :] - line[..., 0, :]   # 1, n, 2
            f0 = line[..., 0, :] - origins   # m, n, 2
            f1 = line[..., 1, :] - origins   # m, n, 2
            # r.shape = 1, 1, 
            sign = (torch.norm(f0, dim=-1, keepdim=True) - r <= 0) | (torch.norm(f1, dim=-1, keepdim=True) - r <= 0)
            sign_mask = ((d * f1).sum(dim=-1, keepdim=True) >= 0) & ((-d * f0).sum(dim=-1, keepdim=True) >= 0)

            s = torch.abs((f0[..., 0] * f1[..., 1] - f0[..., 1] * f1[..., 0])).unsqueeze(-1)
            h = s / torch.norm(d, dim=-1, keepdim=True)
            sign |= (sign_mask & (h - r <= 0))
            is_collision |= sign.squeeze(-1).any(dim=-1, keepdim=False) 
            # a = torch.sum(d * d, dim=-1)
            # b = 2 * torch.sum(f * d, dim=-1)
            # c = torch.sum(f * f, dim=-1) - r.squeeze(-1) ** 2
            # sign = (b**2 - 4*a*c) >= 0

        # if is_collision[1]:
        #     print(self.agent.pos[1])
        #     # print(self.line[11])
        #     collision_lines = torch.nonzero(sign, as_tuple=True)[1]
        #     print("Collision with line indices:", collision_lines.tolist())  # 打印发生碰撞的线的索引
        #     print("Collision with line!!!")

        if len(self.map.triangle_point_array) != 0:
            triangle = self.triangle_points.unsqueeze(0)
            A, B, C = triangle[..., 0, :], triangle[..., 1, :], triangle[..., 2, :]
            v0, v1, v2 = C - A, B - A, origins - A

            dot00 = torch.sum(v0 * v0, dim=-1, keepdim=True)
            dot01 = torch.sum(v0 * v1, dim=-1, keepdim=True)
            dot02 = torch.sum(v0 * v2, dim=-1, keepdim=True)
            dot11 = torch.sum(v1 * v1, dim=-1, keepdim=True)
            dot12 = torch.sum(v1 * v2, dim=-1, keepdim=True)

            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            w = 1 - u - v
            
            sign = ((u >= 0) & (v >= 0) & (w >= 0)).squeeze(-1)
            is_collision |= sign.any(dim=-1, keepdim=False) 
        
        return is_collision
    
    def get_image_pixels(self, w:int=None):
        """
        w(int): 输入为相机分辨率w，默认为agent的分辨率

        return
            origins: shape = (batch_size, 1, 2) 位置信息，防止重复计算
            pixels: shape = (batch_size, w, 2) 相片上每个像素所在的物理坐标
        """
        origins = self.agent.pos.unsqueeze(1) # B, 1, 2
        orientations = self.agent.ori_vector.unsqueeze(1) # B, 1, 2
        if not w:
            w = self.agent.w
        f = self.agent.f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # 1, 1, 1
        field = self.agent.field.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # 1, 1, 1
        field = torch.tan(field * 0.5) # 1, 1, 1
        t = torch.linspace(0, 1, w, device=self.device).unsqueeze(0) # 1, w
        v = torch.stack([orientations[..., 1], -orientations[..., 0]], dim=-1) # B, 1, 2
        v = f * field * v # B, 1, 2
        d = origins + f * orientations - v # B, 1, 2
        pixels = d + t.unsqueeze(-1) * 2 * v # B, w, 2
        return origins, pixels
    
    def get_images(self, w=None):
        """

        """
        origins, pixels = self.get_image_pixels(w) # origins: B, 1, 2 pixels: B, w, 2
        directions = pixels - origins # B, w, 2
        #print()
        d_norm = torch.norm(directions, dim=-1, keepdim=False) # B, w
        img = None
        directions = directions.unsqueeze(2) # B, w, 1, 2
        if len(self.map.circle_center_array) != 0:
            circle_center = self.circle_center.unsqueeze(0) # 1, M, 2
            circle_radius = self.circle_radius.unsqueeze(0) # 1, M, 1

            delta = (origins - circle_center).unsqueeze(1) # B, 1, M, 2
            
            a = (directions ** 2).sum(dim=-1, keepdim=True) # B, w, 1, 1
            b = 2 * (directions * delta).sum(dim=-1, keepdim=True) # B, w, M, 1
            c = (delta ** 2).sum(dim=-1, keepdim=True) - circle_radius ** 2 # B, w, M, 1
            
            discriminant = b ** 2 - 4 * a * c # B, w, M, 1
            valid = discriminant >= 0 # B, w, M, 1
            
            sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0)) # B, w, M, 1
            t1 = (-b - sqrt_discriminant) / (2 * a) # B, w, M, 1
            t2 = (-b + sqrt_discriminant) / (2 * a) # B, w, M, 1

            t = torch.where((t1 >= 0) & valid, t1, t2) # B, w, M, 1
            t = torch.where(valid & (t >= 0), t, torch.tensor(float('inf'), device=self.device)) # B, w, M, 1
            img, _ = torch.min(t.squeeze(-1), dim=-1) # B, w
        directions = directions.squeeze(1) # B, w, 2

        if len(self.map.line_array) != 0:
            line = torch.stack(self.map.line_array).to(self.device) # N, 2, 2
            line = line.unsqueeze(0) # 1, N, 2, 2
            m, n = directions.shape[1], line.shape[1] # m, n = w, N
            if img is None:
                img = torch.full((self.batch_size, m), float('inf'), device=self.device) 
            # print(f"m:{m}, n:{n}")
            # print("Shape of direction", directions.shape)
            e = (line[..., 1, :] - line[..., 0, :]).unsqueeze(0).expand(self.batch_size, m, n, 2) # B, w, N, 2
            b = (line[..., 1, :] - origins).unsqueeze(1).expand(self.batch_size, m, n, 2) # B, w, N, 2
            A = torch.stack([directions.expand(self.batch_size, m, n, 2), e], dim=4) # B, w, N, 4
            valid = torch.abs(torch.det(A)) > 0# B, w
            
            x = torch.full((self.batch_size, m, n, 2), float('inf'), device=self.device) # B, w, N, 2
            if valid.any():
                idx = valid.nonzero(as_tuple=True)
                A, b = A[idx], b[idx]
                x[idx] = torch.linalg.solve(A, b) + 0.0
                t = torch.full((self.batch_size, m, n), float('inf'), device=self.device)
                idx = (x[..., 0] >= 1) & (x[..., 1] >= 0) & (x[..., 1] <= 1)
                t[idx] = x[idx][..., 0]
                t, _ = torch.min(t, dim=2)
                img = torch.where((img<t), img, t)
        # 返回 深度图， 成像线段上像素点实际坐标
        imgs = d_norm * img
        imgs = torch.where(imgs == float('inf'), self.cfg.agent_cfg.field_radius, imgs)
        # print(imgs[0].shape)
        
        return imgs, pixels
    
    def get_map_grid(self, map:Map):
        """
            生成地图的网格坐标，并判断哪些点位于障碍物内部。

            参数:
                map (Map): 输入的地图对象，包含地图尺寸、分辨率及障碍物信息。

            返回:
                tuple:
                    - points (Tensor): 所有网格坐标点 (N, 2)。
                    - points[~mask] (Tensor): 未被障碍物占据的自由空间网格坐标。
                    - points[mask] (Tensor): 位于障碍物内部的网格坐标。
                    - mask (Tensor): 布尔掩码，指示哪些网格点位于障碍物内。
                    - grid (Tensor): 以地图坐标系表示的网格坐标，形状为 (W, H, 2)。
                    - grid_mask (Tensor): 以地图坐标系表示的布尔掩码，形状为 (W, H)，标识障碍物区域。

            过程:
                1. 计算网格尺寸 (H, W) 并生成所有网格坐标点。
                2. 初始化掩码 mask，标识是否位于障碍物内部。
                3. 处理圆形障碍物：
                - 计算网格点到所有圆心的欧几里得距离。
                - 依据半径判断哪些点在圆形障碍物内，更新 mask。
                4. 处理三角形障碍物：
                - 计算点是否落在某个三角形内部，更新 mask。
                5. 转换网格坐标到地图坐标系，并返回相关数据。

            """
        ratio = map.ratio
        H, W = int(map.height/ratio), int(map.width/ratio)
        center = torch.tensor([W//2, H//2], dtype=torch.float32, device=self.device)
        x = torch.arange(W, device=self.device).view(W, 1).expand(W, H)
        y = torch.arange(H, device=self.device).view(1, H).expand(W, H)
        points = torch.stack([x, y], dim=-1).to(torch.float32).reshape(-1, 2)
        mask = torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)
        
        if len(map.circle_center_array) != 0:
            circle_center = torch.stack(map.circle_center_array).to(self.device) / ratio
            circle_radius = torch.stack(map.circle_radius_array).to(self.device) / ratio
            circle_center = trans_simple(circle_center, center)
            d = torch.cdist(points, circle_center)
            in_circle = (d <= circle_radius.T).any(dim=1)
            mask |= in_circle
        
        if len(map.triangle_point_array) != 0:
            triangle = torch.stack(map.triangle_point_array).to(self.device) / ratio
            triangle = trans_simple(triangle, center)

            A, B, C = triangle[..., 0, :], triangle[..., 1, :], triangle[..., 2, :]
            v0 = C - A
            v1 = B - A

            dot00 = torch.einsum('md,md->m', v0, v0)
            dot01 = torch.einsum('md,md->m', v0, v1)
            dot11 = torch.einsum('md,md->m', v1, v1)
            denom = dot00 * dot11 - dot01 * dot01

            v2 = points[:, None, :] - A[None, :, :]
            dot02 = torch.einsum('nmd,md->nm', v2, v0)  
            dot12 = torch.einsum('nmd,md->nm', v2, v1)  
            u = (dot02 * dot11 - dot12 * dot01) / denom
            v = (dot12 * dot00 - dot02 * dot01) / denom

            in_triangle = (u >= 0) & (v >= 0) & (u + v <= 1)
            in_triangle = in_triangle.any(dim=-1)
            mask |= in_triangle

        grid = trans_simple_back(points.clone(), center).reshape(W, H, 2) * ratio
        # print("Shape of grid:", grid.shape)
        grid_mask = mask.reshape(W, H)
        return points, points[~mask], points[mask], mask, grid, grid_mask
    
    def get_img_field(self):
        W, H = self.grid_map.shape[0], self.grid_map.shape[1]
        origins = self.agent.pos.unsqueeze(1) # B, 1, 2
        B = origins.shape[0]
        imgs, pixels = self.get_images(self.w_gt) # B, wgt  B, wgt, 2
        field_radius = self.agent.cfg.field_radius
        imgs = torch.where(imgs == float('inf'), field_radius, imgs)
        # points = self.points_all.clone().unsqueeze(0).expand(B, -1, -1) # B, W*H, 2
        points = self.grid_map.view(1, W * H, 2).repeat(B, 1, 1)
        #points = points.reshape(B, self.W, self.H, 2)
        #points = torch.flip(points, dims=[2]) # B, W, H, 2
        
        vp = pixels - origins
        ap = torch.atan2(vp[..., 1], vp[..., 0]) # B, wgt

        vc = points - origins # B, W*H, 2
        ac = torch.atan2(vc[..., 1], vc[..., 0]) # B, W*H
        dc = torch.norm(vc, dim=-1) # B, W*H

        r = torch.where(dc == 0, torch.tensor(1e-6, device=self.device), dc)
        cosc = torch.sum(vc * self.agent.ori_vector.unsqueeze(1), dim=-1)
        cosc /= r
        cosA = torch.cos(self.agent.field * 0.5)
        mask = (cosc >= cosA) & (r <= field_radius) # B, W*H

        a_diff = torch.abs(ap.unsqueeze(2) - ac.unsqueeze(1))
        pixel_idx = a_diff.argmin(dim=1) # B, W*H

        depths = torch.gather(imgs, 1, pixel_idx) # B, W*H

        mask &= (dc <= depths)
        
        #mask = mask.reshape(B, self.W, self.H)
        #mask = torch.flip(mask, dims=[2]) # B, W, H, 2
        mask_grid = mask.reshape(B, W, H)
        mask_points = mask
        return mask_points, mask_grid

    def update_grid_mask_visible(self):
        delta_point_mask_visible, delta_grid_mask_visible = self.get_img_field()
        # print(self.points_visible.shape, delta_point_mask_visible.shape)
        self.points_visible |= delta_point_mask_visible
        self.grid_mask_visible |= delta_grid_mask_visible
        return
    
    def update_grid_mask_visible_(self):
        # 获取当前视野范围内的网格点布尔张量
        # B, W, H
        delta_grid_mask_visible = self.get_img_field()

        # print(self.grid_mask_visible.shape, delta_grid_mask_visible.shape)
        # 更新视野内的点

        self.grid_mask_visible |= delta_grid_mask_visible
        #self.grid_mask_visible = self.grid_mask_visible | delta_grid_mask_visible

        # 删除其中self.grid_mask_obstacle为True的点
        # B, W, H
        #self.grid_mask_visible = self.grid_mask_visible & (~self.grid_mask_obstacle)

    def generate_safe_pos(self):
        """
        从self.points_safe中随机选取batch_size个初始点，返回形状为(batch_size, 2)的张量。
        """
        
        num_safe_points = self.points_safe.shape[0]
        # 随机生成batch_size个索引（允许重复选取）
        indices = torch.randint(low=0, high=num_safe_points, size=(self.batch_size,))
        # 根据索引从self.points_safe中选取对应的点作为初始位置
        return self.points_safe[indices]


    def generate_desire_pos(self, init_pos):
        """
        对于从 generate_safe_pos 生成的初始点，找到 self.points_safe 中距离其最远的点，
        返回形状为 (batch_size, 2) 的张量。
        """
        
        
        # 计算 self.points_safe 中所有点到每个初始点的欧几里得距离
        # 距离矩阵的形状为 (batch_size, num_safe_points)
        dists = torch.cdist(init_pos, self.points_safe)
        
        # 对于每个初始点，找到距离最远的安全点的索引
        max_indices = torch.argmax(dists, dim=1)
        
        # 选取对应的点作为目标点
        return self.points_safe[max_indices]

    def get_nearest_obstacle_vector(self):
        """
        计算每个智能体到最近障碍物的距离向量。
        """
        if self.points_obstacle.shape[0] == 0:
            raise ValueError("No obstacle points available.")
        
        # 计算所有智能体与所有障碍物点的欧几里得距离
        dists = torch.cdist(self.agent.pos.unsqueeze(0), self.points_obstacle.unsqueeze(0)).squeeze(0)  # (batch_size, num_obstacles)
        
        # 找到最近障碍物的索引
        min_indices = torch.argmin(dists, dim=1)  # (batch_size,)
        
        # 计算距离向量（障碍物位置 - 智能体位置）
        nearest_obstacle_positions = self.points_obstacle[min_indices]  # (batch_size, 2)
        distance_vectors = nearest_obstacle_positions - self.agent.pos  # (batch_size, 2)
        
        return distance_vectors  # 返回每个智能体到最近障碍物的距离向量

    def get_invisible_direction_vector(self):
        """
        返回:
        direction_vector_normalized: torch.Tensor, shape (B, 2)，归一化的二维向量，
                                    分量分别为 (horizontal, vertical)。
        """
        B, H, W = self.batch_size, self.H, self.W
        device = self.device

        # 第一次调用时定义 x_idx 与 y_idx，并保存为类的属性；后续直接使用已有属性
        if not hasattr(self, 'x_idx'):
            self.x_idx = torch.arange(W, device=device).view(1, W, 1)
        if not hasattr(self, 'y_idx'):
            self.y_idx = torch.arange(H, device=device).view(1, 1, H)
        x_idx = self.x_idx
        y_idx = self.y_idx

        # 将每个智能体的位置扩展为 (B, 1, 1)
        pos_idx = self.physical_to_matrix(self.agent.pos)  # (B, 2)
        agent_x = pos_idx[:, 0].clone().view(B, 1, 1)
        agent_y = pos_idx[:, 1].clone().view(B, 1, 1)

        # 分别构造左右、上下区域的 mask
        # 左侧区域：所有 x 坐标小于 agent_x
        left_mask = (x_idx < agent_x)         # (B, W, 1) 自动广播到 (B, W, H)
        # 右侧区域：所有 x 坐标大于 agent_x
        right_mask = (x_idx > agent_x)
        # 上方区域：所有 y 坐标大于 agent_y
        up_mask = (y_idx > agent_y)           # (B, 1, H)
        # 下方区域：所有 y 坐标小于 agent_y
        down_mask = (y_idx < agent_y)

        # 分别对四个方向内的访问次数求和
        up_sum    = (self.grid_visit_time * up_mask).sum(dim=(1, 2))      # (B,)
        down_sum  = (self.grid_visit_time * down_mask).sum(dim=(1, 2))
        left_sum  = (self.grid_visit_time * left_mask).sum(dim=(1, 2))
        right_sum = (self.grid_visit_time * right_mask).sum(dim=(1, 2))

        # 利用启发式：访问次数越少，则期望越大；这里采用 1/(次数+1)（避免除零）
        up_desire    = 1.0 / (up_sum + 1.0)
        down_desire  = 1.0 / (down_sum + 1.0)
        left_desire  = 1.0 / (left_sum + 1.0)
        right_desire = 1.0 / (right_sum + 1.0)

        up_desire = up_desire * (self.W // 2 - self.agent.pos[:, 0])
        down_desire = down_desire * (self.agent.pos[:, 0] + self.W // 2)
        right_desire = left_desire * (self.H // 2 - self.agent.pos[:, 1])
        left_desire = right_desire * (self.agent.pos[:, 1] + self.H // 2)
        #print(up_desire[0], down_desire[0], left_desire[0], right_desire[0])

        vertical   = up_desire - down_desire  # (B,)
        horizontal = right_desire - left_desire   # (B,)

        direction_vector = torch.stack([horizontal, vertical], dim=1)  # (B, 2)

        # 归一化（避免除零）
        norm = torch.norm(direction_vector, dim=1, keepdim=True)
        direction_vector_normalized = torch.where(norm > 0, direction_vector / norm, direction_vector)

        return direction_vector_normalized

    def update_acc_attacc(self):
        """
        为每个智能体生成当前时刻加速度和角加速度
        由两个分量合并：
        1. 从 agent.pos 指向 agent.desired_pos，模长为 sqrt(0.1 * max_acc)
        2. obstacle_opp_vector，模长为 sqrt(2 * 0.5 * max_acc * safe_radius)
        """
        max_acc = torch.tensor(self.cfg.agent_cfg.max_acc, device=self.device)
        safe_radius = self.cfg.agent_cfg.safe_radius
        
        # 计算目标方向向量
        direction_vector = self.agent.desired_pos - self.agent.pos  # (B, 2)
        direction_norm = torch.norm(direction_vector, dim=1, keepdim=True) + 1e-8
        normalized_direction_vector = direction_vector / direction_norm
        # print("Normalized direction vector:", normalized_direction_vector[0])
        
        # 计算目标方向的加速度
        acc_magnitude_goal = torch.sqrt(0.2 * max_acc)
        acc_goal = normalized_direction_vector * acc_magnitude_goal  # (B, 2)
        
        # 计算障碍物反方向向量，并归一化
        obstacle_opp_vector =  -self.get_nearest_obstacle_vector()  # (B, 2)
        obstacle_distance = torch.norm(obstacle_opp_vector, dim=1, keepdim=True) + 1e-8
        normalized_obstacle_opp_vector = obstacle_opp_vector / obstacle_distance
        
        # 计算避障加速度
        acc_magnitude_obstacle = torch.sqrt(2 * max_acc * 1 * safe_radius)
        avoid_obstacle_vector = normalized_obstacle_opp_vector * acc_magnitude_obstacle  # (B, 2)

        # 在智能体到障碍物距离大于 safe_radius 时，将避障加速度设为 0
        mask_safe = obstacle_distance > safe_radius
        avoid_obstacle_vector[mask_safe.squeeze()] = 0

        # 计算总加速度
        
        acc = acc_goal + avoid_obstacle_vector  # (B, 2)
        #print(acc_goal[0], avoid_obstacle_vector[0], acc[0])
        self.agent.acc = torch.clamp(acc, min=-max_acc, max=max_acc)

        # 为每个智能体生成角加速度
        mask_change = self.agent.att_acc_timer >= self.agent.att_acc_change_time
        # print(self.agent.att_acc[mask_change].shape, ((torch.rand((mask_change.sum()), device=self.device) * 2 - 1) * self.cfg.agent_cfg.max_att_acc).shape)
        # $$$$$ Do not change att_acc by now
        self.agent.att_acc[mask_change] = (torch.rand((mask_change.sum()), device=self.device) * 2 - 1) * self.cfg.agent_cfg.max_att_acc
        self.agent.att_acc.zero_()
        self.agent.att_acc_timer[mask_change] = 0

        return
    
    def get_desired_pos(self):
        return self.agent.desired_pos
    
    def get_rand_visible_points(self, N:int):
        """
        作用是从每个Batch中采样N个visible的点
        返回一个shape为3,B,N的tensor indices
        indices[0], indices[1], indices[2] 分别代表 batch_idx, x, y
        用例:
        points = self.grid_map.repeat(B, 1, 1, 1)
        v = points[indices[0], indices[1], indices[2], :]
        返回抽样点的物理坐标
        测试方法:
        v = self.grid_mask_visible[indices[0], indices[1], indices[2]]
        print(torch.all(v).item())
        """
        assert N > 0, 'error number'
        W, H, B = self.W, self.H, self.batch_size
        mask = self.grid_mask_visible.view(B, -1)
        K = mask.sum(dim=1, keepdim=True)
        probs = mask.float() / K
        idx = torch.multinomial(probs, N, replacement=False)
        x = idx // H
        y = idx % H
        batch_idx = torch.arange(B).view(-1, 1).expand(-1, N).to(self.device)
        indices = torch.stack([batch_idx, x, y])
        return indices
    
    def get_local_visible_boundary(self, N: int=20):
        """
        角度均匀采样，idx shape为B,2代表每个batch选出N个点的index，配合batch_idx可直接采点，见函数末尾注释用例
        """
        W, H, B = self.W, self.H, self.batch_size

        points = self.grid_map.unsqueeze(0) # 1, H, W, 2
        origins = self.agent.pos.unsqueeze(1).unsqueeze(1) # B, 1, 1, 2

        dp = points - origins
        dx, dy = dp[..., 0], dp[..., 1]
        d = dx**2 + dy**2 # B, H, W
        d = d.unsqueeze(1).expand(-1, N, -1, -1) # B, N, H, W
        
        theta = torch.atan2(dy.float(), dx.float()) # B, H, W
        theta = (theta + 2 * torch.pi) % (2 * torch.pi) # B, H, W

        n = (theta / (2 * torch.pi / N)).long() % N # B, H, W
        dir_mask = (n.unsqueeze(1) == torch.arange(N, device=self.device).view(1, N, 1, 1))
        
        invisible_mask = ~self.grid_mask_visible.unsqueeze(1)
        valid_mask = dir_mask & invisible_mask

        d_masked = torch.where(valid_mask, d, torch.tensor(float('inf'), device=self.device))
        min_d, flat_indices = torch.min(d_masked.view(B, N, -1), dim=2)
        
        x = flat_indices // W
        y = flat_indices % W

        batch_idx = torch.arange(B).unsqueeze(1).expand(B, N)
        idx = torch.stack([x, y], dim=-1)
        # 这块代码用来判断边界点是未知还是障碍物
        # batch_idx = torch.arange(B).unsqueeze(1).expand(B, N)
        # r, c = idx[..., 0], idx[..., 1]
        # gmo = self.grid_mask_obstacle.unsqueeze(0).expand(2, W, H)
        # v = gmo[batch_idx, r, c]
        # print(v)

        # 这块代码用来判断选到的点是否都是未知点
        # batch_idx = torch.arange(B).unsqueeze(1).expand(B, N)
        # r, c = idx[..., 0], idx[..., 1]
        # v = self.grid_mask_visible[batch_idx, r, c]
        # print(v)

        #print(idx.shape)
        return batch_idx, idx

    def generate_local_ground_truth(self, N:int):
        """
            param:
                N (int): 采样点数目

            return:
                dict:
                    - batch_idx, row_idx, col_idx (Tensor): 用来在shape为B,W,H,...中的tensor里采样
                    - is_obstacle (Tensor): 未被障碍物占据的自由空间网格坐标。
                    - coord (Tensor): 采样点物理坐标
                    - dist (Tensor): 采样点相对距离的平方
                    - theta (Tensor): 采样点相对方向
                    - attitude_encode (Tensor): 采样点相对方向编码
                    
            "batch_idx": batch_idx, # B, N
            "row_idx": r, # B, N
            "col_idx": c, # B, N
            "is_obstacle": v, # B, N
            "coord": coord, # B, N, 2
            "dist": dist, # B, N
            "ori": ori, # B, N
            "attitude_encode": att_encode # B, N, 4 
        """
        assert N > 0, "num error"
        batch_idx, idx = self.get_local_visible_boundary(N=N)
        r, c = idx[..., 0], idx[..., 1]
        gmo = self.grid_mask_obstacle.unsqueeze(0).expand(self.batch_size, self.W, self.H)
        
        v = gmo[batch_idx, r, c]

        origins = self.agent.pos.unsqueeze(1) # B, 1, 2
        coord = self.grid_map.unsqueeze(0).expand(self.batch_size, self.W, self.H, 2) # B, W, H, 2
        coord = coord[batch_idx, r, c, :]
        relative_coord = coord - origins # B, N, 2
        dx, dy = relative_coord[..., 0], relative_coord[..., 1]
        dist = dx**2 + dy**2 # B, N
        ori = torch.atan2(dy.float(), dx.float()) # B, N
        att_encode = attitude_encode(ori.reshape(-1), device=self.device).reshape(self.batch_size, N, -1)
        return {
            "batch_idx": batch_idx, # B, N
            "row_idx": r, # B, N
            "col_idx": c, # B, N
            "is_obstacle": v, # B, N
            "coord": coord, # B, N, 2
            "dist": dist, # B, N
            "ori": ori, # B, N
            "attitude_encode": att_encode # B, N, 4 
        }
    
    def generate_global_ground_truth(self, square_size: int):
        """
        随机选取一个给定大小的正方形，计算已知点比总面积作为探索度供网络训练

        参数:
            square_size (int): 正方形的边长。单位为像素

        返回:
            center_coords (torch.Tensor): 正方形中心点的位置坐标，形状为 (B, 2)。
            center_coords_encoded (torch.Tensor): 正方形中心点的编码，形状为 (B, 12)。
            ground_truth (torch.Tensor): 每个正方形的探索度，形状为 (B, )。
        """
        B, H, W = self.batch_size, self.H, self.W
        device = self.device

        # 生成单个中心点坐标（所有 batch 共享）
        center_x = torch.randint(low=square_size // 2, high=W - square_size // 2, size=(1,), device=device)
        center_y = torch.randint(low=square_size // 2, high=H - square_size // 2, size=(1,), device=device)
        center_coords = torch.stack([center_x, center_y], dim=1)  # (1, 2)

        # 计算正方形区域的索引范围
        x_start, x_end = center_x - square_size // 2, center_x + square_size // 2
        y_start, y_end = center_y - square_size // 2, center_y + square_size // 2

        visible_region = self.grid_mask_visible[:, y_start:y_end, x_start:x_end]  # (B, s, s)

        # 计算每个 batch 的可见点数量
        visible_count = visible_region.reshape(B, -1).sum(dim=1)  # (B, )

        # 计算探索度（可见点数 / 区域总像素数）
        ground_truth = visible_count.float() / (square_size * square_size)  # (B, )

        # 将中心点复制到 batch 维度
        center_coords = center_coords.expand(B, -1)  # (B, 2)
        center_coord_encoded = position_encode(center_coords, device=device)


        return center_coords, center_coord_encoded, ground_truth
    
    def generate_ground_truth(self, square_size: int):
        """
        调用generate_global_ground_truth和generate_local_ground_truth函数
        """
        # 生成全局的 ground truth
        center_coords, center_coord_encoded, global_gt = self.generate_global_ground_truth(self.cfg.agent_cfg.global_quary_square_size)

        # 生成局部的 ground truth
        local_gt = self.generate_local_ground_truth(self.cfg.agent_cfg.local_query_num)

        gt = {}
        gt["global_explrate"] = global_gt.unsqueeze(-1) # B, 1
        gt["global_query_encode"] = center_coord_encoded
        gt["local_query_encode"] = local_gt["attitude_encode"]
        gt["local_gt_distance"] = local_gt["dist"].unsqueeze(-1) # B, N, 1
        gt["local_gt_obstacle"] = local_gt["is_obstacle"].long()

        gt["local_gt_coord"] = local_gt["coord"]

        return gt
        

    def tmp_generate_ground_truth(self, square_size: int):
        """
        随机选取一个给定大小的正方形，根据 self.grid_mask_obstacle 和 self.grid_mask_visible，
        并行计算并返回正方形中心点的位置坐标和正方形中每一个点是否可见可达的信息。

        参数:
            square_size (int): 正方形的边长。单位为像素

        返回:
            center_coords (torch.Tensor): 正方形中心点的位置坐标，形状为 (B, 2)。
            ground_truth (torch.Tensor): 每个正方形中点的可见可达信息，形状为 (B, square_size, square_size)。
                                        0 表示不可见，1 表示可达，2 表示不可达。
        """
        B, H, W = self.batch_size, self.H, self.W
        device = self.device

        # 生成单个中心点坐标（所有 batch 共享）
        center_x = torch.randint(low=square_size // 2, high=W - square_size // 2, size=(1,), device=device)
        center_y = torch.randint(low=square_size // 2, high=H - square_size // 2, size=(1,), device=device)
        center_coords = torch.stack([center_x, center_y], dim=1)  # (1, 2)

        # 计算正方形区域的索引范围
        x_start, x_end = center_x - square_size // 2, center_x + square_size // 2
        y_start, y_end = center_y - square_size // 2, center_y + square_size // 2

        # print(self.grid_mask_obstacle.shape, self.grid_mask_visible.shape)
        # 直接切片访问正方形区域
        visible_region = self.grid_mask_visible[:, y_start:y_end, x_start:x_end]  # (B, s, s)
        obstacle_region = self.grid_mask_obstacle[y_start:y_end, x_start:x_end]  # (B, s, s)

        # 计算 ground_truth
        ground_truth = torch.zeros((B, square_size, square_size), dtype=torch.int, device=device)
        ground_truth[(visible_region & ~obstacle_region)] = 1  # 可见且可达
        ground_truth[(visible_region & obstacle_region)] = 2  # 可见但不可达

        ground_truth = ground_truth.reshape(B, -1)
        coords_encoded = position_encode(center_coords, device=device)

        # print("ground_truth shape:", ground_truth.shape)
        return coords_encoded, ground_truth
        

    def step(self):
        self.update_acc_attacc()
        self.agent.step()
        #self.update_grid_mask_visible_()
        self.update_grid_mask_visible()


        mask_reset = self.is_collision()
        idx_reset = torch.nonzero(mask_reset, as_tuple=True)[0]
        self.reset_idx(idx_reset)

        
        gt = self.generate_ground_truth(self.cfg.agent_cfg.global_quary_square_size)

        step_output = {}
        step_output["image"], _ = self.get_images()
        step_output["agent_pos_encode"] = pos_encode(self.agent.pos, self.agent.ori, device=self.device)
        step_output["gt"] = gt
        step_output["idx_reset"] = idx_reset

        # _, _ = self.get_local_visible_boundary(N=40)

        return step_output

    def reset(self, change_map=False):
        if change_map:
            self.map.random_initialize()
            self.init_grid()
        
        idx = torch.arange(self.batch_size, dtype=torch.int64, device=self.device)
        self.reset_idx(idx)
        return 

    def reset_idx(self, idx):
        if len(idx):
            init_pos = self.generate_safe_pos()
            self.agent.reset_idx(idx, init_pos, self.generate_desire_pos(init_pos))
            # 更新已知点mask
            self.grid_mask_visible[idx] = 0
            self.grid_visit_time[idx] = 0

            self.update_grid_mask_visible()
        return