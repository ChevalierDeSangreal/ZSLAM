import torch
from torch import nn
from camera import Camera
from map import Map
from environment import Environment
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    c = Camera(
        f=math.sqrt(2.)*0.5, 
        theta=math.pi*0.25,
        field=math.pi*0.5,
        w=3,
        x=0,
        y=0,
        L=10.,
    )
    map = Map(10., 10.)
    map.add_circle(1., 3., 1.)
    map.add_circle(3., 1., 1.)
    map.add_line(2., 2., 4., 2.)
    map.add_line(2., 2., 2., 4.)
    map.add_line(4., 2., 2., 4.)
    env = Environment(
        map=map, 
        camera=c, 
        resolution_ratio=0.01,
        device="cuda:0")
    #print(c.get_image_pixels())