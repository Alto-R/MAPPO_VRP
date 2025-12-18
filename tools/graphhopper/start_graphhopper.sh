#!/usr/bin/env bash
# GraphHopper 启动脚本 - 城市快递小卡车配置
# Usage: ./start_graphhopper.sh [import|server]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAR_FILE="$SCRIPT_DIR/graphhopper-web-10.2.jar"
CONFIG_FILE="$SCRIPT_DIR/config.yml"
GRAPH_CACHE="$SCRIPT_DIR/graph-cache"

# 检查 JAR 文件是否存在
if [ ! -f "$JAR_FILE" ]; then
    echo "[ERROR] GraphHopper JAR not found: $JAR_FILE"
    exit 1
fi

# 检查 Java 是否可用
if ! command -v java >/dev/null 2>&1; then
    echo "[ERROR] Java not found. Please install Java 11+ and add to PATH."
    exit 1
fi

# 获取操作参数
ACTION="$1"

# 默认操作：有图则 server，无图则 import
if [ -z "$ACTION" ]; then
    if [ -d "$GRAPH_CACHE" ]; then
        ACTION="server"
    else
        ACTION="import"
    fi
fi

cd "$SCRIPT_DIR"

if [ "$ACTION" = "import" ]; then
    echo "============================================"
    echo "  GraphHopper Import - 构建路网图"
    echo "============================================"
    echo
    echo "[INFO] 正在导入 OSM 数据并构建路网图..."
    echo "[INFO] 这可能需要几分钟时间，请耐心等待..."
    echo

    # 如果存在旧的图缓存，先删除
    if [ -d "$GRAPH_CACHE" ]; then
        echo "[INFO] 清理旧的图缓存..."
        rm -rf "$GRAPH_CACHE"
    fi

    java -Xmx4g -Xms1g -jar "$JAR_FILE" import "$CONFIG_FILE"

    echo
    echo "[SUCCESS] 路网图构建完成！"
    echo "[INFO] 图数据保存在: $GRAPH_CACHE"
    echo
    echo "下一步: 运行 \"./start_graphhopper.sh server\" 启动服务"

elif [ "$ACTION" = "server" ]; then
    echo "============================================"
    echo "  GraphHopper Server - 路由服务"
    echo "============================================"
    echo

    if [ ! -d "$GRAPH_CACHE" ]; then
        echo "[ERROR] 图缓存不存在，请先运行导入:"
        echo "        ./start_graphhopper.sh import"
        exit 1
    fi

    echo "[INFO] 启动 GraphHopper 服务..."
    echo "[INFO] 服务地址: http://localhost:8989"
    echo "[INFO] API 文档: http://localhost:8989/maps/"
    echo "[INFO] 按 Ctrl+C 停止服务"
    echo

    java -Xmx2g -Xms512m -jar "$JAR_FILE" server "$CONFIG_FILE"

else
    echo "Usage: ./start_graphhopper.sh [import|server]"
    echo
    echo "Commands:"
    echo "  import  - 导入 OSM 数据并构建路网图 (首次运行必须执行)"
    echo "  server  - 启动 GraphHopper 路由服务"
    echo
    echo "Example:"
    echo "  ./start_graphhopper.sh import    # 首次运行"
    echo "  ./start_graphhopper.sh server    # 启动服务"
    echo "  ./start_graphhopper.sh           # 自动检测 (无图则import，有图则server)"
fi
