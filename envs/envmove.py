import torch
from camera import Camera
from map import Map
from utils.plot import bool_tensor_visualization
from utils.geometry import get_circle_points, get_line_points, transformation, transformation_back
from utils import theta_to_orientation_vector, batch_theta_to_rotation_matrix
from typing import List
import math
from cfg import *

class Agent:
    def __init__(self, init_pos, agent_cfg:AgentCfg, batch_size, ori=None, device='cpu'):
        """
        初始化无人机代理 配置相机参数 配置无人机参数 配置无人机位置和朝向

        参数:
            init_pos torch.tensor[batch_size, 2]: 相机位置的初始 xy 坐标
            agent_cfg (CameraCfg): 相机配置参数，包括焦距、视场角、图像宽度等。
            ori (float): 相机朝向角度（弧度制），若未提供，则随机初始化。
            batch_size (int): 批大小，即同时模拟的相机数量。
            device (str): 计算设备，默认为 'cpu'。

        主要属性:
            f (torch.tensor): 相机焦距。
            field (torch.tensor): 视场角（弧度制），要求在 (0, π) 范围内。
            w (int): 图像的像素宽度。
            field_radius (torch.tensor): 视场半径。
            init_pos (torch.tensor）: 相机位置的初始 xy 坐标。
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
        
        self.batch_size = batch_size
        self.device = device
        
        # 将所有浮点数转换为 tensor.float 类型
        self.f = torch.tensor(agent_cfg.f, dtype=torch.float, device=device)
        self.field = torch.tensor(agent_cfg.field, dtype=torch.float, device=device)
        self.w = agent_cfg.w
        self.field_radius = torch.tensor(agent_cfg.field_radius, dtype=torch.float, device=device)
        
        self.init_pos = init_pos
        self.pos = init_pos
        # 随机初始化ori（若未提供）
        self.ori = torch.tensor(ori, dtype=torch.float, device=device).expand(batch_size) if ori is not None else torch.rand(batch_size, dtype=torch.float, device=device) * 2 * math.pi

        # 计算旋转矩阵和方向向量
        self.R = batch_theta_to_rotation_matrix(self.ori)
        self.ori_vector = theta_to_orientation_vector(self.ori)
        
        # 计算安全半径
        self.safe_radius = torch.tensor(agent_cfg.safe_radius, dtype=torch.float, device=device) if agent_cfg.safe_radius is not None else self.f / torch.sin(0.5 * self.field)

    
class EnvMove:
    def __init__(
            self, 
            batch_size:int,
            resolution_ratio=0.0,
            device="cpu",
            ):
        # resolution_ratio < 0 不渲染
        self.batch_size = batch_size
        self.device = device

        self.cfg = EnvMoveCfg()

        init_pos = self.get_init_pos()
        self.agent = Agent(init_pos, self.cfg.agent_cfg, batch_size, device=device)

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
        
        self.map_center = torch.tensor([self.H//2, self.W//2], dtype=torch.float32, device=self.device)
        if len(self.map.circle_center_array) != 0:
            self.circle_center = torch.stack(self.map.circle_center_array).to(self.device)
            self.circle_radius = torch.stack(self.map.circle_radius_array).to(self.device)
        if len(self.map.line_array) != 0:
            self.line = torch.stack(self.map.line_array).to(self.device)
        if len(self.map.triangle_point_array) != 0:
            self.triangle_points = torch.stack(self.map.triangle_point_array).to(self.device)
        points_all, points_no_obstacle, points_obs, _ = self.get_map_grid(self.map)

        # tnnd，这里需要用transformation_back是因为从(0, 0)(W, H)平移到(-W/2, -H/2)(W/2, H/2)的坐标系
        # 同时在这个坐标系下，y轴是向下的，而在原坐标系下，y轴是向上的
        self.points_all = transformation_back(points_all, self.map_center) * ratio # 所有点的xy坐标
        self.points_no_obstacle = transformation_back(points_no_obstacle, self.map_center) * ratio # 无障碍物的点的xy坐标
        self.points_obstacle = transformation_back(points_obs, self.map_center) * ratio # 障碍物的点的xy坐标
        self.safe_set = transformation_back(self.get_safe_set(), self.map_center) * ratio # 安全集合的点的xy坐标
        
        self.map_visible = torch.zeros(self.batch_size, self.W, self.H)

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
        _, safe_points, _, _ = self.get_map_grid(safe_map)
        return safe_points


    def is_collision(self):
        # origins = torch.stack([camera.position.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        # r = torch.tensor([camera.safe_radius for camera in self.cameras], device=self.device).unsqueeze(-1).unsqueeze(-1)
        origins = self.agent.pos
        r = self.agent.safe_radius.unsqueeze(-1).unsqueeze(-1)
        is_collision = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        if len(self.map.circle_center_array) != 0:
            distance = torch.norm(self.circle_center-origins, dim=-1, keepdim=True)
            sign = (distance - r - self.circle_radius <  0)
            is_collision |= sign.squeeze(-1).any(dim=-1, keepdim=False)

        if len(self.map.line_array) != 0:
            line = self.line.unsqueeze(0)   
            d = line[..., 1, :] - line[..., 0, :]
            f = line[..., 0, :] - origins
            a = torch.sum(d * d, dim=-1)
            b = 2 * torch.sum(f * d, dim=-1)
            c = torch.sum(f * f, dim=-1) - r.squeeze(-1) ** 2
            sign = (b**2 - 4*a*c) >= 0
            is_collision |= sign.any(dim=-1, keepdim=False) 

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

    def get_image_pixels(self):
        # origins = torch.stack([camera.position.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        # orientations = torch.stack([camera.orientation.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        origins = self.agent.pos
        orientations = self.agent.ori_vector
        w = self.agent.w
        # f = torch.tensor([camera.f for camera in self.cameras], device=self.device).unsqueeze(-1).unsqueeze(-1)
        # field = torch.tensor([camera.field for camera in self.cameras], device=self.device).unsqueeze(-1).unsqueeze(-1)
        f = self.agent.f.unsqueeze(-1).unsqueeze(-1)
        field = self.agent.field.unsqueeze(-1).unsqueeze(-1)
        field = torch.tan(field * 0.5)
        t = torch.linspace(0, 1, w, device=self.device).unsqueeze(0)
        v = torch.stack([orientations[..., 1], -orientations[..., 0]], dim=-1)
        v = f * field * v
        d = origins + f * orientations - v
        pixels = d + t.unsqueeze(-1) * 2 * v
        return origins, pixels
    
    def get_images(self):
        origins, pixels = self.get_image_pixels()
        directions = pixels - origins
        d_norm = torch.norm(directions, dim=-1, keepdim=False)
        img = None
        if len(self.map.circle_center_array) != 0:
            circle_center = self.circle_center.unsqueeze(0)
            circle_radius = self.circle_radius.unsqueeze(0)

            directions = directions.unsqueeze(2)
            delta = (origins - circle_center).unsqueeze(1)
            
            a = (directions ** 2).sum(dim=-1, keepdim=True)
            b = 2 * (directions * delta).sum(dim=-1, keepdim=True)
            c = (delta ** 2).sum(dim=-1, keepdim=True) - circle_radius ** 2
            directions = directions.squeeze(1)
            
            discriminant = b ** 2 - 4 * a * c
            valid = discriminant >= 0
            
            sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0))
            t1 = (-b - sqrt_discriminant) / (2 * a)
            t2 = (-b + sqrt_discriminant) / (2 * a)

            t = torch.where((t1 >= 0) & valid, t1, t2)
            t = torch.where(valid & (t >= 0), t, torch.tensor(float('inf'), device=self.device))
            img, _ = torch.min(t.squeeze(-1), dim=-1)

        if len(self.map.line_array) != 0:
            line = torch.stack(self.map.line_array).to(self.device)
            line = line.unsqueeze(0)
            m, n = directions.shape[1], line.shape[1]
            
            e = (line[..., 1, :] - line[..., 0, :]).unsqueeze(0).expand(self.batch_size, m, n, 2)
            b = (line[..., 1, :] - origins).unsqueeze(1).expand(self.batch_size, m, n, 2)
            A = torch.stack([directions.expand(self.batch_size, m, n, 2), e], dim=4)
            valid = torch.abs(torch.det(A)) >= 1e-6
            
            x = torch.full((self.batch_size, m, n, 2), float('inf'), device=self.device)
            if valid.any():
                idx = valid.nonzero(as_tuple=True)
                A, b = A[idx], b[idx]
                x[idx] = torch.linalg.solve(A, b) + 0.0
                t = torch.full((self.batch_size, m, n), float('inf'), device=self.device)
                idx = (x[..., 0] >= 1) & (x[..., 1] >= 0) & (x[..., 1] <= 1)
                t[idx] = x[idx][..., 0]
                t, _ = torch.min(t, dim=2)
                img = torch.where((img<t), img, t)
        # 返回 深度图，比例图（比例小于1则视为物体在成像平面和相机点之间）
        return d_norm * img, img
    
    def get_map_grid(self, map:Map):
        """
        生成给定地图的网格坐标，并判断哪些点位于障碍物内部。

        参数:
            map (Map): 输入的地图对象，包含地图尺寸、分辨率及障碍物信息。

        返回:
            tuple: 
                - grid[~mask] (Tensor): 自由空间中的网格坐标（未被障碍物占据的点）。
                - grid[mask] (Tensor): 位于障碍物内部的网格坐标。
                - mask (Tensor): 一个布尔掩码，指示每个网格点是否位于障碍物内。
        """
        ratio = map.ratio
        H, W = int(map.height/ratio), int(map.width/ratio)
        center = torch.tensor([H//2, W//2], dtype=torch.float32, device=self.device)
        y = torch.arange(H, device=self.device).view(H, 1).expand(H, W)
        x = torch.arange(W, device=self.device).view(1, W).expand(H, W)
        grid = torch.stack([y, x], dim=-1).to(torch.float32).reshape(-1, 2)
        mask = torch.zeros(grid.shape[0], dtype=torch.bool, device=self.device)
        
        if len(map.circle_center_array) != 0:
            circle_center = torch.stack(map.circle_center_array).to(self.device) / ratio
            circle_radius = torch.stack(map.circle_radius_array).to(self.device) / ratio
            circle_center = transformation(circle_center, center)
            d = torch.cdist(grid, circle_center)
            in_circle = (d <= circle_radius.T).any(dim=1)
            mask |= in_circle
        
        if len(map.triangle_point_array) != 0:
            triangle = torch.stack(map.triangle_point_array).to(self.device) / ratio
            triangle = transformation(triangle, center)

            A, B, C = triangle[..., 0, :], triangle[..., 1, :], triangle[..., 2, :]
            v0 = C - A
            v1 = B - A

            dot00 = torch.einsum('md,md->m', v0, v0)
            dot01 = torch.einsum('md,md->m', v0, v1)
            dot11 = torch.einsum('md,md->m', v1, v1)
            denom = dot00 * dot11 - dot01 * dot01

            v2 = grid[:, None, :] - A[None, :, :]
            dot02 = torch.einsum('nmd,md->nm', v2, v0)  
            dot12 = torch.einsum('nmd,md->nm', v2, v1)  
            u = (dot02 * dot11 - dot12 * dot01) / denom
            v = (dot12 * dot00 - dot02 * dot01) / denom

            in_triangle = (u >= 0) & (v >= 0) & (u + v <= 1)
            in_triangle = in_triangle.any(dim=-1)
            mask |= in_triangle

        return grid, grid[~mask], grid[mask], mask
    
    def get_img_field(self):
        """
        生成相机视野范围内的网格点布尔张量。

        返回:
            in_img_field (Tensor[batch_size, number of no obstacle points]): 布尔张量，表示哪些网格点位于相机的视野范围内。
            True 表示在视野内，False 表示不在视野内。
        """
        points = self.points_no_obstacle
        radius = self.agent.field_radius
        # origins = torch.stack([camera.position.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        # orientations = torch.stack([camera.orientation.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        origins = self.agent.pos
        orientations = self.agent.ori_vector
        vector = points - origins
        product = (orientations * vector).sum(dim=-1, keepdim=True)
        vector_norm = torch.norm(vector, dim=-1, keepdim=True)
        cos = product / (vector_norm + 1e-6)

        field = self.agent.field
        field = torch.cos(field * 0.5).reshape(-1, 1, 1)

        in_img_field = (field <= cos) & (vector_norm <= radius)

        return in_img_field
    
    def step(self):
        pass

    def reset(
            self,
            change_map=False):
        if change_map:
            return
        idx = self.reset_idx()

        if idx.any():
            # 随机初始化
            random_theta = torch.rand(3, device=self.device) * (2 * torch.pi)
            self.agent.ori[idx] = random_theta[idx]
            random_theta = theta_to_orientation_vector(random_theta.unsqueeze(-1).unsqueeze(-1))
            self.agent.ori_vector[idx] = random_theta[idx]

            random_idx = torch.randperm(self.safe_set.shape[0])[:self.batch_size]
            self.agent.pos[idx] = self.safe_set[random_idx][idx]

    def reset_idx(self):
        idx = self.is_collision()
        return idx
    
    def get_init_pos(self):
        return None