import math

class AgentCfg:
    f = math.sqrt(2.)*0.5 # 相机焦距
    ori = None # 相机朝向角度（弧度制）
    field = math.pi*0.5 # 视场角（弧度制），必须在 (0, π) 之间
    w = 4 # 图像的像素宽度
    safe_radius = 0.5 # 安全半径，若未提供，则计算 `f / sin(0.5 * field)`
    field_radius = 100. # 视场半径，默认为 100.0

class MapCfg:
    width = 10. # 地图宽度
    height = 6. # 地图高度
    ratio = 0.1 # 地图分辨率
    max_coverage = 0.1 # 地图最大覆盖率（0到1之间），以地图总面积的比例计算。


class EnvMoveCfg:
    agent_cfg: AgentCfg = AgentCfg()
    map_cfg: MapCfg = MapCfg()
