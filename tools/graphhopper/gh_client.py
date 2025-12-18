"""
GraphHopper API 客户端 - 用于 MAPPO VRP 项目

提供与 GraphHopper 路由服务的接口，获取真实路网的路径规划结果。

Usage:
    from mappo.tools.graphhopper.gh_client import GraphHopperClient

    client = GraphHopperClient()
    route = client.route(
        start=(113.2644, 23.1291),  # 广州市中心 (经度, 纬度)
        end=(113.3245, 23.1067)     # 目的地
    )
    print(f"距离: {route['distance']}m, 时间: {route['time']}ms")
"""

import json
import urllib.request
import urllib.parse
from typing import Tuple, List, Dict, Any, Optional


class GraphHopperClient:
    """GraphHopper 路由服务客户端"""

    def __init__(self, base_url: str = "http://localhost:8995"):
        """
        初始化客户端

        Args:
            base_url: GraphHopper 服务地址
        """
        self.base_url = base_url.rstrip("/")
        self.route_endpoint = f"{self.base_url}/route"

    def is_available(self) -> bool:
        """检查 GraphHopper 服务是否可用"""
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        profile: str = "truck",
        points_encoded: bool = False,
        instructions: bool = False,
        calc_points: bool = True,
    ) -> Dict[str, Any]:
        """
        计算两点之间的路由

        Args:
            start: 起点坐标 (经度, 纬度)
            end: 终点坐标 (经度, 纬度)
            profile: 路由配置文件 ("small_truck", "car", "car_shortest")
            points_encoded: 是否编码路径点 (False 返回原始坐标)
            instructions: 是否返回导航指令
            calc_points: 是否返回路径点

        Returns:
            路由结果字典，包含:
            - distance: 距离 (米)
            - time: 时间 (毫秒)
            - points: 路径点列表 (如果 calc_points=True)
            - bbox: 边界框
        """
        params = {
            "point": [f"{start[1]},{start[0]}", f"{end[1]},{end[0]}"],
            "profile": profile,
            "points_encoded": str(points_encoded).lower(),
            "instructions": str(instructions).lower(),
            "calc_points": str(calc_points).lower(),
        }

        # 构建 URL
        query_parts = []
        for key, value in params.items():
            if isinstance(value, list):
                for v in value:
                    query_parts.append(f"{key}={urllib.parse.quote(str(v))}")
            else:
                query_parts.append(f"{key}={urllib.parse.quote(str(value))}")

        url = f"{self.route_endpoint}?{'&'.join(query_parts)}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

                if "paths" not in data or len(data["paths"]) == 0:
                    raise ValueError("No route found")

                path = data["paths"][0]
                return {
                    "distance": path.get("distance", 0),  # 米
                    "time": path.get("time", 0),  # 毫秒
                    "points": path.get("points", {}).get("coordinates", []),
                    "bbox": path.get("bbox", []),
                    "raw": path,
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"GraphHopper API error: {e.code} - {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot connect to GraphHopper: {e.reason}")

    def matrix(
        self,
        points: List[Tuple[float, float]],
        profile: str = "truck",
        out_arrays: List[str] = None,
    ) -> Dict[str, Any]:
        """
        计算多点之间的距离/时间矩阵

        Args:
            points: 坐标点列表 [(经度, 纬度), ...]
            profile: 路由配置文件
            out_arrays: 输出数组类型 ["distances", "times", "weights"]

        Returns:
            矩阵结果字典，包含:
            - distances: 距离矩阵 (米)
            - times: 时间矩阵 (毫秒)
        """
        if out_arrays is None:
            out_arrays = ["distances", "times"]

        # 构建请求体
        request_body = {
            "points": [[lon, lat] for lon, lat in points],
            "profile": profile,
            "out_arrays": out_arrays,
        }

        url = f"{self.base_url}/matrix"
        data = json.dumps(request_body).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"GraphHopper Matrix API error: {e.code} - {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot connect to GraphHopper: {e.reason}")


def test_connection():
    """测试 GraphHopper 连接"""
    print("=" * 50)
    print("GraphHopper 连接测试")
    print("=" * 50)

    client = GraphHopperClient()

    # 检查服务是否可用
    print("\n[1] 检查服务状态...")
    if not client.is_available():
        print("    [FAIL] GraphHopper 服务不可用")
        print("    请先运行: start_graphhopper.bat server")
        return False
    print("    [OK] 服务运行中")

    # 测试路由 - 广州市内两点
    print("\n[2] 测试路由计算...")
    try:
        # 广州天河区到越秀区
        start = (113.3245, 23.1291)  # 天河
        end = (113.2644, 23.1291)    # 越秀

        route = client.route(start, end, profile="truck")

        distance_km = route["distance"] / 1000
        time_min = route["time"] / 1000 / 60

        print(f"    起点: {start}")
        print(f"    终点: {end}")
        print(f"    距离: {distance_km:.2f} km")
        print(f"    时间: {time_min:.1f} 分钟")
        print(f"    路径点数: {len(route['points'])}")
        print("    [OK] 路由计算成功")
    except Exception as e:
        print(f"    [FAIL] 路由计算失败: {e}")
        return False

    # 测试矩阵 API
    print("\n[3] 测试距离矩阵...")
    try:
        points = [
            (113.2644, 23.1291),  # 点1
            (113.3245, 23.1067),  # 点2
            (113.3500, 23.1200),  # 点3
        ]

        matrix = client.matrix(points, profile="truck")

        print(f"    点数: {len(points)}")
        print(f"    距离矩阵 (km):")
        for i, row in enumerate(matrix.get("distances", [])):
            row_km = [f"{d/1000:.1f}" for d in row]
            print(f"      {i}: {row_km}")
        print("    [OK] 矩阵计算成功")
    except Exception as e:
        print(f"    [WARN] 矩阵 API 可能不可用: {e}")

    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_connection()
