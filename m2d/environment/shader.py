import torch
from torch import nn
from map.map import Map
from camera.camera import Camera
import matplotlib.pyplot as plt

class Shader:
    def __init__(self, map:Map):
        self.map = map
        pass

    def visualize(self, scale=8):
        plt.figure(figsize=(self.map.width / scale, self.map.height / scale))
        plt.imshow(self.map.map.cpu(), cmap='gray', origin='upper')
        plt.axis('off')
        plt.show()

