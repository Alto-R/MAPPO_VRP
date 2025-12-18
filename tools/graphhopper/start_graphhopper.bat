@echo off
REM GraphHopper 启动脚本 - 城市快递小卡车配置
REM Usage: start_graphhopper.bat [import|server]

setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set JAR_FILE=%SCRIPT_DIR%graphhopper-web-10.2.jar
set CONFIG_FILE=%SCRIPT_DIR%config.yml
set GRAPH_CACHE=%SCRIPT_DIR%graph-cache

REM 检查 JAR 文件是否存在
if not exist "%JAR_FILE%" (
    echo [ERROR] GraphHopper JAR not found: %JAR_FILE%
    exit /b 1
)

REM 检查 Java 是否可用
java -version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Java not found. Please install Java 11+ and add to PATH.
    exit /b 1
)

REM 默认操作为 server (如果图已存在) 或 import (如果不存在)
set ACTION=%1
if "%ACTION%"=="" (
    if exist "%GRAPH_CACHE%\nodes" (
        set ACTION=server
    ) else (
        set ACTION=import
    )
)

cd /d "%SCRIPT_DIR%"

if "%ACTION%"=="import" (
    echo ============================================
    echo   GraphHopper Import - 构建路网图
    echo ============================================
    echo.
    echo [INFO] 正在导入 OSM 数据并构建路网图...
    echo [INFO] 这可能需要几分钟时间，请耐心等待...
    echo.

    REM 如果存在旧的图缓存，先删除
    if exist "%GRAPH_CACHE%" (
        echo [INFO] 清理旧的图缓存...
        rmdir /s /q "%GRAPH_CACHE%"
    )

    java -Xmx4g -Xms1g -jar "%JAR_FILE%" import "%CONFIG_FILE%"

    if errorlevel 1 (
        echo [ERROR] 导入失败！
        exit /b 1
    )

    echo.
    echo [SUCCESS] 路网图构建完成！
    echo [INFO] 图数据保存在: %GRAPH_CACHE%
    echo.
    echo 下一步: 运行 "start_graphhopper.bat server" 启动服务

) else if "%ACTION%"=="server" (
    echo ============================================
    echo   GraphHopper Server - 路由服务
    echo ============================================
    echo.

    if not exist "%GRAPH_CACHE%\nodes" (
        echo [ERROR] 图缓存不存在，请先运行导入:
        echo         start_graphhopper.bat import
        exit /b 1
    )

    echo [INFO] 启动 GraphHopper 服务...
    echo [INFO] 服务地址: http://localhost:8990
    echo [INFO] API 文档: http://localhost:8990/maps/
    echo [INFO] 按 Ctrl+C 停止服务
    echo.

    java -Xmx2g -Xms512m -jar "%JAR_FILE%" server "%CONFIG_FILE%"

) else (
    echo Usage: start_graphhopper.bat [import^|server]
    echo.
    echo Commands:
    echo   import  - 导入 OSM 数据并构建路网图 (首次运行必须执行)
    echo   server  - 启动 GraphHopper 路由服务
    echo.
    echo Example:
    echo   start_graphhopper.bat import    # 首次运行
    echo   start_graphhopper.bat server    # 启动服务
    echo   start_graphhopper.bat           # 自动检测 (无图则import，有图则server)
)

endlocal
