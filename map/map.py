import torch
import random
import json
import math
from cfg import MapCfg

class Map:
    def __init__(self, cfg:MapCfg):
        self.ratio = cfg.ratio
        self.width = cfg.width
        self.height = cfg.height
        self.max_coverage = cfg.max_coverage
        self.circle_center_array = []
        self.circle_radius_array = []
        self.safe_center_array = []
        self.safe_radius_array = []
        self.line_array = []
        self.triangle_array = []
        self.triangle_point_array = []
        
    def add_circle(self, x, y, r):
        self.circle_center_array.append(torch.tensor([x, y]))
        self.circle_radius_array.append(torch.tensor([r]))

    def add_line(self, x0, y0, x1, y1):
        self.line_array.append(torch.tensor([
            [x0, y0],
            [x1, y1]
        ]))
    
    def add_triangle(self, p0, p1, p2):
        """
        p0, p1, p2: 三角形的三个顶点坐标, list
        """
        self.line_array.append(torch.tensor([
            p0,
            p1
        ]))
        self.line_array.append(torch.tensor([
            p1,
            p2
        ]))
        self.line_array.append(torch.tensor([
            p2,
            p0
        ]))
        self.triangle_array.append(torch.tensor([
            len(self.line_array)-3, 
            len(self.line_array)-2, 
            len(self.line_array)-1
        ]))
        self.triangle_point_array.append(torch.tensor([
            p0, 
            p1, 
            p2
        ]))

    def add_triangle_idx(self, idx):
        self.triangle_array.append(idx)

    def get_grid_map(self):
        H, W = int(self.height/self.ratio), int(self.width/self.ratio)
        grid = torch.zeros((H, W), dtype=torch.bool)
        if len(self.circle_center_array) != 0:
            pass
        if len(self.triangle_point_array) != 0:
            pass

        pass

    def to_json_data(self):
        data = {
            'width': self.width,
            'height': self.height,
            'ratio': self.ratio,
            'circle_center_array': [center.tolist() for center in self.circle_center_array],
            'circle_radius_array': [radius.tolist() for radius in self.circle_radius_array],
            'safe_center_array': [center.tolist() for center in self.safe_center_array],
            'safe_radius_array': [radius.tolist() for radius in self.safe_radius_array],
            'line_array': [line.tolist() for line in self.line_array],
            'triangle_array': [line for line in self.triangle_array],
            'triangle_point_array': [line.tolist() for line in self.triangle_point_array],
        }
        return data

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        
        map_instance = cls(data['width'], data['height'], data['ratio'])
        for center, radius in zip(data['circle_center_array'], data['circle_radius_array']):
            map_instance.add_circle(center[0], center[1], radius[0])
        # for line in data['line_array']:
        #     map_instance.add_line(line[0][0], line[0][1], line[1][0], line[1][1])
        # for idx in data['triangle_array']:
        #     map_instance.add_triangle_idx(idx)
        for points in data['triangle_point_array']:
            map_instance.add_triangle(
                torch.tensor(points[0]),
                torch.tensor(points[1]),
                torch.tensor(points[2])
            )

        return map_instance
    

    def random_initialize(self):
        """
        根据最大地图覆盖率随机初始化地图。
        
        
        说明：
            随机生成圆形和三角形，这些物体的面积之和累计达到目标覆盖面积后停止。
            允许物体间重叠。
        """
        total_map_area = self.width * self.height
        target_area = self.max_coverage * total_map_area
        current_area = 0.0
        
        # 仅选择具有面积的物体：普通圆、三角形
        shapes = ['circle', 'triangle']
        
        while current_area < target_area:
            shape = random.choice(shapes)
            if shape == 'circle':
                x = random.uniform(0, self.width) - self.width / 2
                y = random.uniform(0, self.height) - self.height / 2
                # 半径范围可根据地图尺寸调整，这里取1到地图短边十分之一之间
                r = random.uniform(min(self.width, self.height) / 50, min(self.width, self.height) / 10)
                self.add_circle(x, y, r)
                current_area += math.pi * r * r
            elif shape == 'triangle':
                # 改进算法：基准点生成三角形
                base_x = random.uniform(0, self.width) - self.width / 2
                base_y = random.uniform(0, self.height) - self.height / 2
                p0 = torch.tensor([base_x, base_y], dtype=torch.float32)
                
                # 限制边长为地图短边的十分之一
                max_edge = min(self.width, self.height) / 10
                d1 = random.uniform(1, max_edge)
                d2 = random.uniform(1, max_edge)
                angle1 = random.uniform(0, 2 * math.pi)
                angle2 = random.uniform(0, 2 * math.pi)
                
                p1 = torch.tensor([
                    base_x + d1 * math.cos(angle1), 
                    base_y + d1 * math.sin(angle1)
                ], dtype=torch.float32)
                
                p2 = torch.tensor([
                    base_x + d2 * math.cos(angle2), 
                    base_y + d2 * math.sin(angle2)
                ], dtype=torch.float32)
                
                # 计算三角形面积
                area_triangle = abs(
                    p0[0].item() * (p1[1].item() - p2[1].item()) +
                    p1[0].item() * (p2[1].item() - p0[1].item()) +
                    p2[0].item() * (p0[1].item() - p1[1].item())
                ) / 2.0
                
                self.add_triangle(p0.tolist(), p1.tolist(), p2.tolist())
                current_area += area_triangle

        # 添加四个狭长的边界三角形
        boundary_height = 1 * self.ratio  # 确保足够窄
        w, h = self.width, self.height

        # 上边界
        self.add_triangle(
            [-w / 2, h / 2], 
            [w / 2, h / 2], 
            [0, h / 2 - boundary_height]
        )

        # 下边界
        self.add_triangle(
            [-w / 2, -h / 2], 
            [w / 2, -h / 2], 
            [0, -h / 2 + boundary_height]
        )

        # 左边界
        self.add_triangle(
            [-w / 2, h / 2], 
            [-w / 2, -h / 2], 
            [-w / 2 + boundary_height, 0]
        )

        # 右边界
        self.add_triangle(
            [w / 2, h / 2], 
            [w / 2, -h / 2], 
            [w / 2 - boundary_height, 0]
        )