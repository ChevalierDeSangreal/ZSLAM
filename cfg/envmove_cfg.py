import math

class AgentCfg:
    f = 0.05 # 相机焦距
    ori = None # 相机朝向角度（弧度制）
    field = math.pi*0.5 # 视场角（弧度制），必须在 (0, π) 之间
    w = 64 # 图像的像素宽度
    safe_radius = f / math.sin(0.5 * field) # 安全半径，若未提供，则计算 `f / sin(0.5 * field)`
    field_radius = 10. # 视场半径，默认为 100.0，单位为m

    max_speed = 1.0 # 最大速度分量
    max_acc = 1.5 # 最大加速度分量
    max_att_acc = math.pi / 3 # 最大角加速度，弧度
    max_att_speed = math.pi / 3 # 最大角速度，弧度

    max_att_acc_change_step = 100 # 角加速度变化最大时间间隔, 单位为step
    min_att_acc_change_step = 10 # 角加速度变化最小时间间隔, 单位为step

    global_quary_square_size = 40 # global gt图像的边长，单位为像素
    local_query_num = 10 # local gt 查询的个数

class MapCfg:
    width = 5. # 地图宽度
    height = 5. # 地图高度
    ratio = 0.1 # 地图分辨率
    max_coverage = 0.05 # 地图最大覆盖率（0到1之间），以地图总面积的比例计算。


class EnvMoveCfg:
    dt = 0.02 # 时间步长
    agent_cfg: AgentCfg = AgentCfg()
    map_cfg: MapCfg = MapCfg()
