import torch
from camera import Camera
from map import Map
from utils.plot import bool_tensor_visualization
from utils.geometry import get_circle_points, get_line_points, transformation, transformation_back
from utils import theta_to_orientation_vector
from typing import List
import math

class Environment:
    def __init__(
            self, 
            map:Map,
            camera:Camera,
            resolution_ratio=0.0,
            device="cpu",
            ):
        # resolution_ratio < 0 不渲染
        self.device = device
        self.camera = camera
        self.map = map
        self.__resolution_ratio = resolution_ratio
        if resolution_ratio > 0:
            H, W = int(self.map.height/resolution_ratio), int(self.map.width/resolution_ratio)
            self.H, self.W = H, W
            self.grid = torch.zeros((H, W), device=device, dtype=torch.bool)
            self.center = torch.tensor([H//2, W//2], dtype=torch.int32, device=device)
            self.init_grid()
            bool_tensor_visualization(self.grid.to("cpu"))
    
    def init_grid(self):
        points = None

        if len(self.map.circle_center_array) != 0:
            circle_center = torch.stack(self.map.circle_center_array).to(self.device) / self.__resolution_ratio
            circle_radius = torch.stack(self.map.circle_radius_array).to(self.device) / self.__resolution_ratio
            points = get_circle_points(circle_center, circle_radius)

        if len(self.map.line_array) != 0:
            line = torch.stack(self.map.line_array).to(self.device) / self.__resolution_ratio
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

    def is_collision(self):
        origins = self.camera.position.to(self.device)
        is_collision = False
        r = self.camera.safe_radius
        if len(self.map.circle_center_array) != 0:
            circle_center = torch.stack(self.map.circle_center_array).to(self.device)
            circle_radius = torch.stack(self.map.circle_radius_array).to(self.device)

            distance = torch.norm(circle_center-origins, dim=-1, keepdim=True)
            is_collision |= ((distance > circle_radius - r) & (distance < circle_radius + r)).any().item()

        if len(self.map.line_array) != 0:
            line = torch.stack(self.map.line_array).to(self.device)
            d0, d1 = torch.norm(line[:, 0, :]-origins, dim=-1, keepdim=True), torch.norm(line[:, 1, :]-origins, dim=-1, keepdim=True)
            is_collision |= (d0 < r) & (d1 < r)
        
        return is_collision


    def get_image(self):
        origins = self.camera.position.to(self.device)
        directions = self.camera.get_image_pixels(self.device) - origins
        d_norm = torch.norm(directions, dim=-1, keepdim=False)
        img = None
        if len(self.map.circle_center_array) != 0:
            circle_center = torch.stack(self.map.circle_center_array).to(self.device)
            circle_radius = torch.stack(self.map.circle_radius_array).to(self.device)

            directions = directions.unsqueeze(1)
            delta = origins - circle_center
            
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
            img, _ = torch.min(t.squeeze(-1), dim=1)

        if len(self.map.line_array) != 0:
            line = torch.stack(self.map.line_array).to(self.device)
            m, n = directions.shape[0], line.shape[0]
            
            e = (line[:, 1, :] - line[:, 0, :]).unsqueeze(0).expand(m, n, 2)
            b = (line[:, 1, :] - origins).unsqueeze(0).expand(m, n, 2)

            A = torch.stack([directions.unsqueeze(1).expand(m, n, 2), e], dim=3)
            valid = torch.abs(torch.det(A)) >= 1e-6
            
            x = torch.full((m, n, 2), float('inf'), device=self.device)
            if valid.any():
                idx = valid.nonzero(as_tuple=True)
                print(len(idx))
                A, b = A[idx], b[idx]
                x[idx] = torch.linalg.solve(A, b) + 0.0
                t = torch.full((m, n), float('inf'), device=self.device)
                idx = (x[..., 0] >= 1) & (x[..., 1] >= 0) & (x[..., 1] <= 1)
                t[idx] = x[idx][..., 0]
                t, _ = torch.min(t, dim=1)
                img = torch.where((img<t), img, t)
        
        return d_norm * img, img
            

class EnvironmentMultiCamera:
    def __init__(
            self, 
            batch_size:int,
            map:Map,
            cameras:List[Camera],
            resolution_ratio=0.0,
            device="cpu",
            ):
        # resolution_ratio < 0 不渲染
        assert batch_size==len(cameras), 'env error'
        self.batch_size = batch_size
        self.device = device
        self.cameras = cameras
        self.cameras = {
            'position': torch.stack([camera.position.unsqueeze(0).to(device) for camera in cameras], dim=0),
            'theta': torch.tensor([camera.theta for camera in cameras], device=device),
            'orientation': torch.stack([camera.orientation.unsqueeze(0).to(device) for camera in cameras], dim=0),
            'w': cameras[0].w,
            'safe_radius': torch.tensor([camera.safe_radius for camera in cameras], device=device),
            'field_radius': torch.tensor([camera.field_radius for camera in cameras], device=device),
            'f': torch.tensor([camera.f for camera in cameras], device=device),
            'field': torch.tensor([camera.field for camera in cameras], device=device)
        }
        self.map = map
        self.__resolution_ratio = resolution_ratio
        self.initial_state = {
            'position': torch.stack([camera.position.unsqueeze(0).to(self.device) for camera in cameras], dim=0),
            'orientation': torch.stack([camera.orientation.unsqueeze(0).to(self.device) for camera in cameras], dim=0)
        }
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
        H, W = int(self.map.height/ratio), int(self.map.width/ratio)
        self.map_center = torch.tensor([H//2, W//2], dtype=torch.float32, device=self.device)
        if len(self.map.circle_center_array) != 0:
            self.circle_center = torch.stack(self.map.circle_center_array).to(self.device)
            self.circle_radius = torch.stack(self.map.circle_radius_array).to(self.device)
        if len(self.map.line_array) != 0:
            self.line = torch.stack(self.map.line_array).to(self.device)
        if len(self.map.triangle_point_array) != 0:
            self.triangle_points = torch.stack(self.map.triangle_point_array).to(self.device)
        points, _, _ = self.get_map_grid(self.map)
        self.points = transformation_back(points, self.map_center) * ratio
        self.safe_set = transformation_back(self.get_safe_set(), self.map_center) * ratio
    
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
        r = torch.max(self.cameras['safe_radius']).item()
        safe_map = Map(
            width=self.map.width,
            height=self.map.height,
            ratio=self.map.ratio
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
        safe_points, _, _ = self.get_map_grid(safe_map)
        return safe_points


    def is_collision(self):
        # origins = torch.stack([camera.position.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        # r = torch.tensor([camera.safe_radius for camera in self.cameras], device=self.device).unsqueeze(-1).unsqueeze(-1)
        origins = self.cameras['position']
        r = self.cameras['safe_radius'].unsqueeze(-1).unsqueeze(-1)
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
        origins = self.cameras['position']
        orientations = self.cameras['orientations']
        w = self.cameras['w']
        # f = torch.tensor([camera.f for camera in self.cameras], device=self.device).unsqueeze(-1).unsqueeze(-1)
        # field = torch.tensor([camera.field for camera in self.cameras], device=self.device).unsqueeze(-1).unsqueeze(-1)
        f = self.cameras['f'].unsqueeze(-1).unsqueeze(-1)
        field = self.cameras['field'].unsqueeze(-1).unsqueeze(-1)
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

        return grid[~mask], grid[mask], mask
    
    def get_img_field(self, radius):
        """
        radius 需要为float或shape为batch_size, 1, 1的tensor
        """
        #radius = torch.tensor([radius for _ in range(self.batch_size)], device=self.device)
        points = self.points
        radius = radius
        # origins = torch.stack([camera.position.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        # orientations = torch.stack([camera.orientation.unsqueeze(0).to(self.device) for camera in self.cameras], dim=0)
        origins = self.cameras['position']
        orientations = self.cameras['orientations']
        vector = points - origins
        product = (orientations * vector).sum(dim=-1, keepdim=True)
        vector_norm = torch.norm(vector, dim=-1, keepdim=True)
        cos = product / (vector_norm + 1e-6)

        field = self.cameras['field']
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
            self.cameras['theta'][idx] = random_theta[idx]
            random_theta = theta_to_orientation_vector(random_theta.unsqueeze(-1).unsqueeze(-1))
            self.cameras['orientation'][idx] = random_theta[idx]

            random_idx = torch.randperm(self.safe_set.shape[0])[:self.batch_size]
            self.cameras['position'][idx] = self.safe_set[random_idx][idx]

    def reset_idx(self):
        idx = self.is_collision()
        return idx
