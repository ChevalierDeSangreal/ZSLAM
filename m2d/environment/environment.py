import torch
from camera import Camera
from map import Map
from utils.plot import bool_tensor_visualization
from utils.geometry import get_circle_points, get_line_points, transformation

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
                A, b = A[idx], b[idx]
                x[idx] = torch.linalg.solve(A, b) + 0.0
                print(x.shape)
                t = torch.full((m, n), float('inf'), device=self.device)
                idx = (x[..., 0] >= 1) & (x[..., 1] >= 0) & (x[..., 1] <= 1)
                t[idx] = x[idx][..., 0]
                print(t)
                t, _ = torch.min(t, dim=1)
                img = torch.where((img<t), img, t)
        
        return d_norm * img
            

