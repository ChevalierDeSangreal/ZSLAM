import torch
from map import Block, DiscBlock, TriangleBlock
from utils import single_theta_to_rotation_matrix, single_theta_to_orientation_vector
import math

class Camera:
    def __init__(self, f, theta, field, w, x, y, L, safe_radius=None, field_radius=100.):
        assert 0 < field < math.pi, 'wrong field angle'
        self.f = f
        self.theta = theta
        self.field = field
        self.R = single_theta_to_rotation_matrix(theta)
        self.orientation = single_theta_to_orientation_vector(theta)
        self.v = torch.tensor([self.orientation[1], -self.orientation[0]])
        self.w = w
        self.x = x
        self.y = y
        self.position = torch.tensor([x, y])
        self.L = L
        self.safe_radius = f / math.sin(0.5 * field)
        self.field_radius = field_radius
        if safe_radius is not None:
             self.safe_radius = max(self.safe_radius, safe_radius)
    
    def move(self, theta=None, x=None, y=None):
        if theta is not None:
            self.theta = theta
            self.R = single_theta_to_rotation_matrix(theta)
            self.orientation = single_theta_to_orientation_vector(theta)
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y

    def get_image_pixels(self, device="cpu"):
        t = torch.linspace(0, 1, self.w, device=device)
        v = self.v.to(device)
        v = self.f * math.tan(self.field * 0.5) * v
        d = self.position.to(device) + self.f * self.orientation.to(device) - v
        pixels = d + t.unsqueeze(1) * 2 * v
        return pixels

