#!/usr/bin/env python3
"""
专业服务启动脚本
启动所有专业模块的MCP服务
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent

# 服务配置
SERVICES = {
    "math": {
        "script": "agent_tools/start_mcp_services.py",
        "port": 8000,
        "env_var": "MATH_HTTP_PORT"
    },
    "stock_local": {
        "script": "agent_tools/tool_get_price_local.py",
        "port": 8003,
        "env_var": "GETPRICE_HTTP_PORT"
    },
    "search": {
        "script": "agent_tools/tool_jina_search.py",
        "port": 8001,
        "env_var": "SEARCH_HTTP_PORT"
    },
    "trade": {
        "script": "agent_tools/tool_trade.py",
        "port": 8002,
        "env_var": "TRADE_HTTP_PORT"
    },
    "portfolio_optimization": {
        "script": "agent_tools/tool_portfolio_optimization.py",
        "port": 8004,
        "env_var": "PORTFOLIO_OPTIMIZATION_HTTP_PORT"
    },
    "multi_timeframe": {
        "script": "agent_tools/tool_multi_timeframe.py",
        "port": 8005,
        "env_var": "MULTI_TIMEFRAME_HTTP_PORT"
    }
}


def start_service(service_name, service_config):
    """启动单个服务"""
    script_path = project_root / service_config["script"]
    port = service_config["port"]
    env_var = service_config["env_var"]
    
    if not script_path.exists():
        print(f"❌ Service script not found: {script_path}")
        return None
    
    # 设置环境变量
    env = os.environ.copy()
    env[env_var] = str(port)
    
    print(f"🚀 Starting {service_name} service on port {port}...")
    
    try:
        # 启动服务进程
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            env=env,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待服务启动
        time.sleep(2)
        
        if process.poll() is None:
            print(f"✅ {service_name} service started successfully on port {port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ {service_name} service failed to start")
            if stdout:
                print(f"   STDOUT: {stdout}")
            if stderr:
                print(f"   STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {service_name} service: {e}")
        return None


def stop_services(processes):
    """停止所有服务"""
    print("\n🛑 Stopping all services...")
    
    for service_name, process in processes.items():
        if process and process.poll() is None:
            print(f"🛑 Stopping {service_name} service...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"✅ {service_name} service stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️  {service_name} service did not stop gracefully, forcing...")
                process.kill()


def main():
    """主函数"""
    print("🎯 AI-Trader Professional Services Launcher")
    print("=" * 50)
    
    # 检查必要的环境变量
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables and try again.")
        return
    
    # 启动服务
    processes = {}
    
    for service_name, service_config in SERVICES.items():
        process = start_service(service_name, service_config)
        if process:
            processes[service_name] = process
        else:
            print(f"❌ Failed to start {service_name}, stopping all services...")
            stop_services(processes)
            return
    
    print("\n🎉 All professional services started successfully!")
    print("Services running:")
    for service_name, service_config in SERVICES.items():
        if service_name in processes:
            print(f"   - {service_name}: http://localhost:{service_config['port']}")
    
    print("\nPress Ctrl+C to stop all services...")
    
    try:
        # 保持运行
        while True:
            time.sleep(1)
            
            # 检查是否有服务停止
            for service_name, process in processes.items():
                if process.poll() is not None:
                    print(f"❌ {service_name} service stopped unexpectedly")
                    stop_services(processes)
                    return
                    
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal...")
        stop_services(processes)
        print("👋 All services stopped. Goodbye!")


if __name__ == "__main__":
    main()
