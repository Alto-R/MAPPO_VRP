# 地理数据加载与道路网络生成

本文档详细说明 `geo_data_loader.py` 模块的设计原理，特别是**卡车路线节点的简化与采样方法**。

## 概述

在真实地理场景（如深圳配送场景）中，卡车必须沿 OSM 道路行驶。然而，完整的道路网络包含成千上万个点，直接作为动作空间会导致：
1. **动作空间爆炸**：无法有效训练强化学习模型
2. **计算开销过大**：每步决策需要考虑太多选项

因此，我们采用**离散化采样**的方法，将连续的道路网络简化为有限数量的**路线节点**。

---

## 道路网络节点生成算法

### 核心思路：放射状采样

从仓库（快递出发点）出发，向多个方向发射"探测路径"，沿每条路径均匀采样点，最终得到覆盖仓库周边道路网络的节点集合。

```
                    N (0°)
                    │
                    │
        NW (315°)   │   NE (45°)
              ╲     │     ╱
               ╲    │    ╱
                ╲   │   ╱
        W (270°) ───●─── E (90°)    ● = 仓库 (Depot)
                ╱   │   ╲
               ╱    │    ╲
              ╱     │     ╲
        SW (225°)   │   SE (135°)
                    │
                    │
                    S (180°)
```

### 采样参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `radius_km` | 10.0 | 采样半径（公里） |
| `num_directions` | 8 | 方向数（N, NE, E, SE, S, SW, W, NW） |
| `points_per_direction` | 5 | 每个方向采样的点数 |

**理论节点数**: `1 (仓库) + 8 × 5 = 41` 个节点

**实际节点数**: 约 30-35 个（去重后），因为：
- 部分方向可能无法到达（水域、山区）
- 相邻方向在靠近仓库处会重叠

---

## 详细算法步骤

### Step 1: 仓库点吸附到道路

```python
center_snapped = self.snap_to_road(center_lon, center_lat)
```

将快递出发点坐标吸附到最近的 OSM 道路上，确保卡车起点在可行驶位置。

**吸附方法**：向 GraphHopper 请求一个极短距离的路由，起点会被自动吸附到最近道路。

### Step 2: 生成目标点

对于每个方向 `i`（0 到 7），计算该方向上距离 `radius_km` 处的目标点：

```python
angle = 2 * π * i / 8   # 方向角（弧度）

# 计算目标点经纬度
target_lon = center_lon + Δlon
target_lat = center_lat + Δlat
```

**角度对应关系**：
| i | 角度 | 方向 |
|---|------|------|
| 0 | 0° | 正北 (N) |
| 1 | 45° | 东北 (NE) |
| 2 | 90° | 正东 (E) |
| 3 | 135° | 东南 (SE) |
| 4 | 180° | 正南 (S) |
| 5 | 225° | 西南 (SW) |
| 6 | 270° | 正西 (W) |
| 7 | 315° | 西北 (NW) |

### Step 3: 请求 GraphHopper 路由

```python
route = gh.route(
    start=center_snapped,    # 仓库（道路上）
    end=(target_lon, target_lat),  # 目标点
    profile="truck",         # 卡车路由
    calc_points=True         # 返回路径点
)
```

GraphHopper 返回**实际可行驶的道路路径**，包含路径上的所有拐点坐标。

### Step 4: 沿路径均匀采样

```python
path_points = route['points']  # 路径点列表
num_points = len(path_points)

if num_points > points_per_direction:
    # 均匀采样 5 个点
    indices = np.linspace(0, num_points - 1, 5, dtype=int)
    sampled = [path_points[i] for i in indices]
else:
    # 点数不足，全部保留
    sampled = path_points
```

**采样示意**：

```
仓库 ──●──────●──────●──────●──────● 目标
      ↑      ↑      ↑      ↑      ↑
     [0]    [1]    [2]    [3]    [4]

路径共有 20 个拐点，均匀取 index = [0, 5, 10, 15, 19]
```

### Step 5: 去除重复点

由于相邻方向的路径在靠近仓库处可能重叠，需要去除距离过近的点：

```python
unique_points = self._remove_duplicate_points(all_points, tolerance_m=50)
```

两点间距离小于 50 米视为重复，只保留一个。

---

## 为什么这样设计？

### 1. 离散动作空间适配强化学习

MAPPO 使用**离散动作空间** `Discrete(N)`，卡车需要从有限选项中选择目标：

```python
# 卡车动作空间
action_space = Discrete(1 + num_route_nodes + 2 * num_drones)
#              ↑    ↑                    ↑
#            STAY  移动到节点X         释放/回收无人机
```

如果使用连续动作空间（直接输出目标坐标），训练难度会显著增加。

### 2. 保证道路可达性

采样点来自**实际路由结果**，确保：
- 所有节点都在 OSM 道路网络上
- 从仓库到任意节点都存在可行路径
- 卡车不会"穿墙"或进入不可行驶区域

### 3. 覆盖度与计算效率平衡

| 方向数 | 每方向点数 | 总节点数（约） | 特点 |
|-------|-----------|--------------|------|
| 4 | 3 | ~12 | 覆盖不足，可能漏掉重要区域 |
| **8** | **5** | **~33** | **平衡选择** |
| 16 | 8 | ~100 | 覆盖充分，但动作空间较大 |

默认参数 `8 × 5` 在覆盖度和训练效率之间取得平衡。

---

## 重要说明

### 卡车可以在任意位置停留

虽然动作空间只包含 33 个目标节点，但**卡车可以在行驶途中的任意位置停留**：

```python
# 动作 0: STAY - 卡车停止移动，保持当前位置
if action == 0:
    truck.velocity = 0
    # 卡车停在当前位置（可能是道路中间）
```

路线节点是**目标点**，不是**必停点**。卡车选择一个目标后，会沿道路向该目标移动，期间可以随时停下。

### 节点数量可配置

通过修改 `generate_road_network_nodes()` 的参数可以调整节点数量：

```python
# 更多节点（更细粒度）
nodes = loader.generate_road_network_nodes(
    center_lon, center_lat,
    radius_km=15.0,        # 更大范围
    num_directions=16,     # 更多方向
    points_per_direction=8 # 每方向更多点
)
# 结果：约 100+ 节点
```

---

## 代码位置

- **主模块**: [geo_data_loader.py](geo_data_loader.py)
- **核心方法**: `GeoDataLoader.generate_road_network_nodes()`
- **使用场景**: [scenarios/shenzhen_delivery.py](scenarios/shenzhen_delivery.py)

## 相关参数

| 命令行参数 | 说明 |
|-----------|------|
| `--road_radius_km` | 道路采样半径，默认 10.0 km |
| `--depot_index` | 快递出发点索引，默认 3（顺丰福田） |
| `--use_graphhopper` | 是否启用 GraphHopper，深圳场景必须为 True |

## 示例输出

```
[GeoDataLoader] Loaded 7480 features
[GeoDataLoader] Found 10 express points
[GeoDataLoader] Found 489 residential points
[GeoDataLoader] Generated 33 road network nodes
[ShenzhenDelivery] Depot: 顺丰速运(深圳福田营业部) @ (114.04234, 22.60910)
[ShenzhenDelivery] Route nodes: 33 (including depot)
[ShenzhenDelivery] Available customers: 180 within 5.0km
```
