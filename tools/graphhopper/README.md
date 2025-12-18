# GraphHopper 路由服务

城市快递小卡车路由服务，基于 GraphHopper 10.2 和广东省 OSM 地图数据。

## 目录结构

```
tools/graphhopper/
├── graphhopper-web-10.2.jar   # GraphHopper 服务 JAR
├── config.yml                 # 配置文件
├── start_graphhopper.bat      # 启动脚本
├── gh_client.py               # Python API 客户端
├── graph-cache/               # 路网图缓存 (自动生成)
└── README.md                  # 本文档
```

## 快速开始

### 1. 启动服务

```batch
cd tools\graphhopper
start_graphhopper.bat server
```

服务启动后：
- Web UI: http://localhost:8990
- API: http://localhost:8990/route

### 2. 重新导入地图 (可选)

如果需要更新地图数据或修改配置后重建：

```batch
start_graphhopper.bat import
```

## Python API 使用

### 基本路由查询

```python
from mappo.tools.graphhopper.gh_client import GraphHopperClient

# 创建客户端
client = GraphHopperClient("http://localhost:8990")

# 检查服务是否可用
if not client.is_available():
    print("请先启动 GraphHopper 服务")
    exit()

# 计算两点间路由 (坐标格式: 经度, 纬度)
route = client.route(
    start=(113.3245, 23.1291),  # 起点: 广州天河
    end=(113.2644, 23.1067),    # 终点: 广州越秀
    profile="truck"             # 使用 truck 配置
)

print(f"距离: {route['distance'] / 1000:.2f} km")
print(f"时间: {route['time'] / 1000 / 60:.1f} 分钟")
print(f"路径点数: {len(route['points'])}")
```

### 距离/时间矩阵

```python
# 计算多点间的距离矩阵
points = [
    (113.2644, 23.1291),  # 点A
    (113.3245, 23.1067),  # 点B
    (113.3500, 23.1200),  # 点C
]

matrix = client.matrix(points, profile="truck")

# 距离矩阵 (米)
distances = matrix["distances"]  # [[0, d_AB, d_AC], [d_BA, 0, d_BC], [d_CA, d_CB, 0]]

# 时间矩阵 (毫秒)
times = matrix["times"]
```

### 运行测试

```batch
cd tools\graphhopper
python gh_client.py
```

## 配置说明

### 可用的路由配置 (Profiles)

| Profile | 说明 |
|---------|------|
| `truck` | 货车配置，考虑限高、限重等限制 |
| `car`   | 标准汽车配置 |

### 修改配置

编辑 `config.yml` 后需要重新导入：

```batch
start_graphhopper.bat import
```

### 主要配置项

```yaml
graphhopper:
  # OSM 数据文件
  datareader.file: ../../data/guangdong-251216.osm.pbf

  # 路由配置
  profiles:
    - name: truck
      custom_model_files: [truck.json]
    - name: car
      custom_model_files: [car.json]

server:
  application_connectors:
    - type: http
      port: 8990          # 服务端口
      bind_host: localhost
```

## REST API

### 路由查询

```
GET /route?point={lat},{lon}&point={lat},{lon}&profile=truck
```

示例：
```
http://localhost:8990/route?point=23.1291,113.3245&point=23.1067,113.2644&profile=truck
```

### 距离矩阵

```
POST /matrix
Content-Type: application/json

{
  "points": [[113.2644, 23.1291], [113.3245, 23.1067]],
  "profile": "truck",
  "out_arrays": ["distances", "times"]
}
```

## 常见问题

### Q: 启动报错 "Invalid or corrupt jarfile"
确保使用正确的 JAR 文件版本 (graphhopper-web-10.2.jar)。

### Q: 导入时内存不足
增加 Java 堆内存，编辑 `start_graphhopper.bat`：
```batch
java -Xmx8g -Xms2g -jar ...
```

### Q: 找不到路由
1. 确认坐标在广东省范围内
2. 检查坐标格式是否正确 (经度, 纬度)
3. 某些偏远地区可能没有道路数据

### Q: 如何更换地图
1. 下载新的 OSM PBF 文件到 `data/` 目录
2. 修改 `config.yml` 中的 `datareader.file` 路径
3. 运行 `start_graphhopper.bat import` 重新导入
