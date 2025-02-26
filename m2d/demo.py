import torch
from torch import nn
from camera import Camera
from map import Map
import environment.environment as Env
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    c = Camera(
        f=math.sqrt(2.)*0.5, 
        theta=math.pi*0.25,
        field=math.pi*0.5,
        w=4,
        x=0,
        y=0,
        L=10.,
    )
    map = Map(10., 10., ratio=0.01)
    map.add_circle(1., 3., 1.)
    map.add_circle(3., 1., 1.)
    map.add_triangle(
        p0=[2., 2.], 
        p1=[2., 4.],
        p2=[4., 2.]
    )
    env = Env.EnvironmentMultiCamera(
        batch_size=3,
        map=map,
        cameras=[c, c, c],
        resolution_ratio=0,
        device="cuda:0"
    )
    #env.get_map_grid()
    env.reset()
    #print(env.get_map_grid().shape)
    
    #env.get_images()
    # env = environment.Environment(
    #     map=map, 
    #     camera=c, 
    #     resolution_ratio=0.01,
    #     device="cuda:0")
    #print(c.get_image_pixels())