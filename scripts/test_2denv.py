import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/wangzimo/VTT/ZSLAM')
from envs import *



def test_circle_map_generation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_envs = 3
    map_r = 5.0
    cm = CircleMap(num_envs=num_envs, device=device, map_r=map_r, num_component=10)
    cm.generate_map()
    
    # 检查每个环境中的每个组件
    for env in range(num_envs):
        components = cm.map_componet[env]
        for comp in components:
            x, y, r = comp
            dist = torch.sqrt(x**2 + y**2)
            assert 0.5 <= dist <= map_r, f"圆心距离错误: {dist.item()}"
            assert r > 0, "半径必须为正"
            assert r <= (dist - 0.5), f"半径超过允许范围: {r.item()} > {dist.item() - 0.5}"

def test_camera_coverage():
    cc = CameraCoverage(fov=math.pi/2)  # 90度视场
    
    # 第一次更新：覆盖 [π/4, 3π/4]
    angles = torch.tensor([math.pi/2])
    cc.update(angles)
    
    # 测试边界条件
    queries = torch.tensor([math.pi/4 - 0.1, math.pi/4, 3*math.pi/4, 3*math.pi/4 + 0.1])
    results = cc.query(queries)
    assert results.tolist() == [0, 1, 1, 0], "基础覆盖测试失败"
    
    # 第二次更新：扩展到 [π/4, 5π/4]
    cc.update(torch.tensor([math.pi]))
    results = cc.query(torch.tensor([math.pi + 0.1]))
    assert results.item() == 1, "连续扩展测试失败"

def test_render_depth():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = TwoDEnv(num_envs=1, device=device)
    
    # 手动设置一个简单场景：原点正前方3米处有一个半径1米的圆
    env.map.map_componet[0] = torch.tensor([[3.0, 0.0, 1.0]], device=device)
    
    # 渲染正前方0度视角
    depth = env.render(cam_angle=0.0, fov=math.pi/180, num_rays=1)
    # print(depth)
    assert torch.allclose(depth, torch.tensor([[2.0]], device=device), atol=1e-3), "深度计算错误"

if __name__ == "__main__":
    # 初始化环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = TwoDEnv(num_envs=3, device=device)
    
    # 生成地图并可视化第一个环境
    env.map.generate_map()
    env.map.plot_map(env_idx=0)
    
    # 进行渲染并可视化结果
    depth = env.render(cam_angle=math.pi/4, fov=math.pi/2, num_rays=100)
    env.visualize_render(depth, cam_angle=math.pi/4, fov=math.pi/2)
    
    # 运行测试用例
    test_circle_map_generation()
    test_camera_coverage()
    test_render_depth()
    print("All tests passed!")