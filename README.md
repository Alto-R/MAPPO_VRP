# MAPPO 卡车-无人机协同配送

基于 Multi-Agent PPO (MAPPO) 算法的多智能体强化学习框架，用于解决卡车-无人机协同配送场景下的车辆路径问题 (VRP)。

## 项目结构

```
mappo/
├── config.py                  # 配置解析器
├── algorithms/                # 算法模块
│   ├── r_mappo/              # MAPPO 算法实现
│   │   ├── r_mappo.py        # 训练器
│   │   └── algorithm/        # 策略与网络
│   │       ├── rMAPPOPolicy.py
│   │       └── r_actor_critic.py
│   └── utils/                # 神经网络组件
│       ├── mlp.py            # MLP 网络
│       ├── cnn.py            # CNN 网络
│       ├── rnn.py            # RNN 网络
│       ├── act.py            # 动作层
│       ├── distributions.py  # 概率分布
│       └── popart.py         # PopArt 归一化
├── utils/                     # 通用工具
│   ├── util.py               # 工具函数
│   ├── separated_buffer.py   # 经验回放缓冲区
│   └── valuenorm.py          # 值函数归一化
├── runner/                    # 训练运行器
│   ├── base_runner.py        # 基类 Runner
│   └── vrp_runner.py         # VRP 专用 Runner
├── envs/                      # 环境模块
│   └── vrp/                   # VRP 环境
│       ├── core.py            # 核心实体类
│       ├── environment.py     # 环境主类
│       ├── VRP_env.py         # 环境工厂函数
│       ├── env_wrappers.py    # 向量化环境包装器
│       └── scenarios/         # 场景配置
├── scripts/                   # 训练脚本
│   └── train_vrp.py           # 训练入口
├── results/                   # 训练结果与模型
└── visualization/             # 可视化工具
```

## 环境概述

VRP 环境模拟**卡车-无人机协同配送**场景：
- **卡车**：作为"母舰"沿预定路线行驶，负责释放和回收无人机
- **无人机**：从卡车起飞，完成客户配送任务后返回卡车充电
- **目标**：在有限时间内，以最小能耗完成所有客户的配送

### 智能体

| 智能体 | 数量 | 动作空间 |
|-------|------|---------|
| Truck | 1 | Discrete(1 + num_nodes + 2 × num_drones) |
| Drone | num_drones | Discrete(2 + num_customers) |

### 动作说明

**卡车动作**：
- `STAY`: 停止移动
- `MOVE_TO_NODE_X`: 前往路线节点 X
- `RELEASE_DRONE_X`: 释放无人机 X
- `RECOVER_DRONE_X`: 回收无人机 X

**无人机动作**：
- `HOVER`: 悬停
- `RETURN_TO_TRUCK`: 返回卡车
- `DELIVER_TO_CUSTOMER_X`: 配送到客户 X

## 快速开始

### 依赖安装

本项目已包含完整的 MAPPO 算法实现，无需外部依赖。安装基础依赖：

```bash
pip install torch numpy wandb setproctitle tensorboardX gym
```

### 训练

#### 基础场景 (抽象环境)

```bash
cd mappo/scripts

# 基础训练
python train_vrp.py --scenario_name truck_drone_basic --num_drones 2 --num_customers 3

# 使用 GPU
python train_vrp.py --cuda --num_drones 2 --num_customers 5

# 完整配置示例
python train_vrp.py \
    --scenario_name truck_drone_basic \
    --algorithm_name mappo \
    --experiment_name test_run \
    --num_drones 2 \
    --num_customers 3 \
    --num_route_nodes 5 \
    --num_env_steps 1000000 \
    --episode_length 200 \
    --n_rollout_threads 8 \
    --use_eval
```

#### 深圳真实地理场景

使用真实 POI 数据和 OSM 道路网络的配送场景。

**前置条件**: 需要启动 GraphHopper 路由服务

```bash
# 1. 先启动 GraphHopper 服务 (在另一个终端)
cd tools/graphhopper
start_graphhopper.bat server

# 2. 训练深圳配送场景
cd mappo/scripts
python train_vrp.py \
    --scenario_name shenzhen_delivery \
    --use_graphhopper \
    --depot_index 3 \
    --num_customers 10 \
    --num_drones 2 \
    --customer_radius_km 5.0 \
    --road_radius_km 10.0 \
    --num_env_steps 1000000 \
    --episode_length 200 \
    --n_rollout_threads 4
```

**可选快递出发点** (`--depot_index`):

| 索引 | 名称 | 坐标 |
|-----|------|------|
| 0 | 八卦岭邮政/综合楼 | (114.094, 22.559) |
| 1 | 明达辉航空快运 | (113.814, 22.619) |
| 2 | 顺丰速运 (宝安) | (113.828, 22.622) |
| 3 | **顺丰速运 (福田)** | (114.042, 22.609) |
| 4 | 顺丰速运 (机场) | (113.816, 22.643) |

### 主要参数

#### 通用参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--scenario_name` | truck_drone_basic | 场景名称 |
| `--algorithm_name` | mappo | 算法 (mappo/rmappo/ippo) |
| `--num_drones` | 2 | 无人机数量 |
| `--num_customers` | 3 | 客户数量 |
| `--episode_length` | 200 | 每回合最大步数 |
| `--num_env_steps` | 1000000 | 总训练步数 |
| `--n_rollout_threads` | 8 | 并行环境数 |
| `--delivery_threshold` | 0.05 | 配送完成距离阈值 |
| `--recovery_threshold` | 0.1 | 无人机回收距离阈值 |

#### 基础场景参数 (truck_drone_basic)

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--num_route_nodes` | 5 | 卡车路线节点数 |

#### 深圳场景参数 (shenzhen_delivery)

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--use_graphhopper` | False | 启用 GraphHopper 道路距离计算 |
| `--graphhopper_url` | http://localhost:8989 | GraphHopper 服务地址 |
| `--geojson_path` | data/poi_...geojson | POI 数据文件路径 |
| `--depot_index` | 3 | 快递出发点索引 (默认福田顺丰) |
| `--customer_radius_km` | 5.0 | 客户筛选半径 (km) |
| `--road_radius_km` | 10.0 | 道路网络生成半径 (km) |

## 奖励机制

| 奖励项 | 值 | 说明 |
|-------|---|------|
| 时间惩罚 | -0.1/步 | 鼓励快速完成 |
| 配送奖励 | +10.0 | 每完成一个客户配送 |
| 完成奖励 | +50.0 | 服务所有客户 |
| 能耗成本 | -0.1 × 耗电 | 无人机能耗 |
| 强制返航惩罚 | -0.5 | 电量不足被迫返航 |
| 未完成惩罚 | -20.0 × 未服务数 | 回合结束时未服务的客户 |

## 训练输出

训练结果保存在 `results/VRP/<scenario>/<algorithm>/<experiment>/` 目录下：

```
run1/
├── models/           # 保存的模型权重
│   ├── actor_agent0.pt
│   ├── critic_agent0.pt
│   └── ...
└── logs/             # TensorBoard 日志
    ├── events.out.tfevents.*
    └── summary.json
```

使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir mappo/results/VRP/truck_drone_basic/mappo/
```

## 环境 API

### 基础场景

```python
from mappo.envs.vrp.VRP_env import VRPEnv
from argparse import Namespace

# 创建配置
args = Namespace(
    scenario_name='truck_drone_basic',
    num_drones=2,
    num_customers=3,
    num_route_nodes=5,
    episode_length=200,
    delivery_threshold=0.05,
    recovery_threshold=0.1,
    use_graphhopper=False
)

# 创建环境
env = VRPEnv(args)

# 环境信息
print(f"智能体数量: {env.n}")                    # 3 (1 truck + 2 drones)
print(f"动作空间: {env.action_space}")           # [Discrete(10), Discrete(5), Discrete(5)]
print(f"观测空间: {env.observation_space}")

# 运行回合
obs_n = env.reset()
for step in range(200):
    actions = [space.sample() for space in env.action_space]
    obs_n, reward_n, done_n, info_n = env.step(actions)
    if all(done_n):
        break

env.close()
```

### 深圳场景 (需要 GraphHopper)

```python
from mappo.envs.vrp.VRP_env import VRPEnv
from argparse import Namespace

# 创建配置
args = Namespace(
    scenario_name='shenzhen_delivery',
    geojson_path='data/poi_batch_1_final_[7480]_combined_5.0km.geojson',
    graphhopper_url='http://localhost:8989',
    depot_index=3,              # 顺丰速运(福田)
    customer_radius_km=5.0,     # 5km内客户
    road_radius_km=10.0,        # 10km道路网络
    num_drones=2,
    num_customers=10,
    episode_length=200,
    delivery_threshold=0.03,
    recovery_threshold=0.05,
    use_graphhopper=True,
    geo_bounds=None             # 自动计算
)

# 创建环境 (需要 GraphHopper 服务运行)
env = VRPEnv(args)

print(f"智能体数量: {env.n}")
print(f"卡车动作空间: {env.action_space[0].n}")   # 1 + 道路节点数 + 2*无人机数
print(f"无人机动作空间: {env.action_space[1].n}") # 2 + 客户数
```

## 算法支持

- **MAPPO**: Multi-Agent PPO，默认算法
- **RMAPPO**: 带 RNN 的 MAPPO
- **IPPO**: 独立 PPO (非中心化 Critic)

注意：由于卡车和无人机有不同的动作空间，本环境强制使用**分离策略模式** (`share_policy=False`)。

## 更多文档

- [VRP 环境详细文档](envs/vrp/claude.md) - 核心类、动作空间、观测空间、奖励函数的详细说明
- [场景设计](envs/vrp/scenarios/claude.md) - 场景配置与自定义方法
- [地理数据与道路网络](envs/vrp/geo_data_loader.md) - 道路节点采样算法、坐标转换详解
