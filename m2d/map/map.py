import torch
import json


class Map:
    def __init__(self, width, height, ratio=0.01):
        self.ratio = ratio
        self.width = width
        self.height = height
        self.circle_center_array = []
        self.circle_radius_array = []
        self.line_array = []
        self.triangle_array = []
        
    def add_circle(self, x, y, r):
        self.circle_center_array.append(torch.tensor([x, y]))
        self.circle_radius_array.append(torch.tensor([r]))

    def add_line(self, x0, y0, x1, y1):
        self.line_array.append(torch.tensor([
            [x0, y0],
            [x1, y1]
        ]))
    
    def add_triangle(self, p0, p1, p2):
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
        self.triangle_array.append([len(self.line_array)-3, len(self.line_array)-2, len(self.line_array)-1])

    def add_triangle_idx(self, idx):
        self.triangle_array.append(idx)

    def to_json_data(self):
        data = {
            'width': self.width,
            'height': self.height,
            'ratio': self.ratio,
            'circle_center_array': [center.tolist() for center in self.circle_center_array],
            'circle_radius_array': [radius.tolist() for radius in self.circle_radius_array],
            'line_array': [line.tolist() for line in self.line_array],
            'triangle_array': [line for line in self.triangle_array],
        }
        return data

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        
        map_instance = cls(data['width'], data['height'], data['ratio'])
        for center, radius in zip(data['circle_center_array'], data['circle_radius_array']):
            map_instance.add_circle(center[0], center[1], radius[0])
        for line in data['line_array']:
            map_instance.add_line(line[0][0], line[0][1], line[1][0], line[1][1])
        for idx in data['triangle_array']:
            map_instance.add_triangle_idx(idx)

        return map_instance
