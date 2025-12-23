#!/usr/bin/env python3
"""
主基准测试程序
协调各个模块完成完整的基准测试流程
"""

import asyncio
import logging
import sys
import os
import traceback
from datetime import datetime

# 获取当前脚本目录和项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 切换到项目根目录，确保所有相对路径正确
os.chdir(project_root)
print(f"Working directory changed to: {os.getcwd()}")

# 添加项目根目录到Python路径
sys.path.insert(0, project_root)

# 导入分离的模块
from run_benchmark.logger_utils import setup_logging
from run_benchmark.argument_parser import parse_and_validate_arguments
from run_benchmark.engine_manager import start_vllm_engine, stop_vllm_engine
from run_benchmark.server_manager import setup_servers, cleanup_servers
from run_benchmark.task_manager import setup_benchmark_tasks, run_benchmark_tasks, cancel_monitor_task
from run_benchmark.result_processor import process_and_save_results

# 导入配置和绘图模块
from config.Config import GLOBAL_CONFIG

try:
    from plot.plotMain import plot_result
except ImportError:
    plot_result = None


async def main():
    """主函数 - 协调各个模块完成基准测试"""
    # 1. 初始化日志系统
    logger = setup_logging()
    
    # 2. 解析和验证参数
    args = parse_and_validate_arguments()
    if args is None:
        logger.error("Parameter validation failed, exiting...")
        return

    logger.info("=== Benchmark Configuration ===")
    logger.info(f"Experiment type: {args.exp}")
    logger.info(f"Short clients: {args.short_clients}, Long clients: {args.long_clients}")
    logger.info(f"Round time: {args.round_time}s, Total rounds: {args.round}")

    # 3. 初始化后端（vLLM引擎或HTTP客户端）
    vllm_engine = None
    
    if getattr(args, 'start_engine', True):
        # 启动vLLM引擎
        logger.info("=== Starting vLLM Engine ===")
        vllm_engine = await start_vllm_engine(args, logger)
        if vllm_engine is None:
            logger.error("Failed to start vLLM engine, exiting...")
            return
        
        # 添加vLLM引擎到全局配置
        GLOBAL_CONFIG['vllm_engine'] = vllm_engine
        logger.info("✓ vLLM engine started successfully")
    else:
        # 使用HTTP客户端连接远程服务
        logger.info("=== Using HTTP Client ===")
        logger.info(f"Connecting to remote vLLM server at: {args.vllm_url}")

    # 4. 设置全局配置
    GLOBAL_CONFIG['round_time'] = args.round_time
    if GLOBAL_CONFIG.get('exp_time', 36000) < args.round_time * args.round:
        GLOBAL_CONFIG['exp_time'] = args.round_time * args.round * 3

    # 5. 设置服务器和连接
    servers = setup_servers(args)

    # 6. 创建结果队列
    all_results = asyncio.Queue()
    request_queue = asyncio.Queue()

    start_time = datetime.now().timestamp()

    try:
        # 7. 设置和运行基准测试任务
        logger.info("Setting up benchmark tasks...")
        tasks, monitor_task, clients, queue_manager = await setup_benchmark_tasks(
            args, all_results, request_queue, logger
        )

        # 8. 运行基准测试
        logger.info("Running benchmark tasks...")
        try:
            await run_benchmark_tasks(tasks, logger)
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

        # 9. 处理和保存结果
        logger.info("Processing and saving results...")
        benchmark_results, total_time, filename, plot_data = process_and_save_results(
            tasks, start_time, args, logger
        )

        # 10. 清理资源
        logger.info("Cleaning up resources...")
        cleanup_servers(servers)
        await cancel_monitor_task(monitor_task, logger)

        # 停止队列管理器（如果存在）
        if queue_manager:
            await queue_manager.stop()
            logger.info("Queue manager stopped")

        # 11. 生成结果图表
        try:
            plot_result(plot_data)
        except ImportError:
            logger.warning("Could not import plot_result, skipping result plotting")

        logger.info(f"Benchmark completed successfully! Results saved to: {filename}")

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        traceback.print_exc()
    finally:
        # 清理后端资源
        if vllm_engine:
            logger.info("Stopping vLLM engine...")
            stop_vllm_engine(vllm_engine, logger)
            logger.info("✓ vLLM engine cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
