#!/usr/bin/env python3
"""
引擎管理模块
处理vLLM引擎的启动、停止和配置
"""

import asyncio
import logging
import os
import subprocess
import time
import signal
import psutil
import pynvml
from config.Config import GLOBAL_CONFIG

# 导入vLLM相关模块
try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncEngineDeadError
    vllm_available = True
except ImportError:
    vllm_available = False


def setup_vllm_logging(log_level="WARNING", suppress_engine_logs=True):
    """
    设置vLLM相关的日志级别
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        suppress_engine_logs: 是否抑制引擎请求/完成相关的详细日志
    """
    import logging
    
    # 设置vLLM相关的logger
    vllm_loggers = [
        'vllm',
        'vllm.async_llm_engine',
        'vllm.engine.async_llm_engine',
        'vllm.core.scheduler',
        'vllm.worker.model_runner',
        'vllm.executor.gpu_executor',
        'vllm.model_executor.model_loader'
    ]
    
    log_level_num = getattr(logging, log_level.upper(), logging.WARNING)
    
    for logger_name in vllm_loggers:
        vllm_logger = logging.getLogger(logger_name)
        vllm_logger.setLevel(log_level_num)
        
        # 如果要抑制引擎日志，特别处理async_llm_engine
        if suppress_engine_logs and 'async_llm_engine' in logger_name:
            # 添加一个过滤器来屏蔽特定消息
            class EngineLogFilter(logging.Filter):
                def filter(self, record):
                    # 屏蔽"Added request"和"Finished request"消息
                    message = record.getMessage()
                    if ("Added request" in message or 
                        "Finished request" in message or
                        "Aborted request" in message):
                        return False
                    return True
            
            vllm_logger.addFilter(EngineLogFilter())


# 全局变量存储引擎进程和监控任务
vllm_process = None
queue_monitor_task = None


async def monitor_engine_queue(engine, interval=5, logger=None):
    """
    监控引擎队列状态的异步任务
    
    Args:
        engine: vLLM引擎实例
        interval: 监控间隔（秒）
    """
    logger.info(f"开始监控引擎队列状态，间隔{interval}秒")
    
    while True:
        try:
            # 使用正确的scheduler访问方法
            if hasattr(engine, 'engine') and hasattr(engine.engine, 'scheduler'):
                scheduler_list = engine.engine.scheduler
                
                # scheduler是一个list，获取第一个调度器对象
                if isinstance(scheduler_list, list) and len(scheduler_list) > 0:
                    scheduler = scheduler_list[0]
                    
                    # 获取队列信息
                    waiting_queue_size = len(scheduler.waiting) if hasattr(scheduler, 'waiting') else 0
                    running_queue_size = len(scheduler.running) if hasattr(scheduler, 'running') else 0
                    swapped_queue_size = len(scheduler.swapped) if hasattr(scheduler, 'swapped') else 0
                    
                    # 获取总的未完成请求数
                    total_unfinished = 0
                    if hasattr(scheduler, 'get_num_unfinished_seq_groups'):
                        total_unfinished = scheduler.get_num_unfinished_seq_groups()
                    
                    # 获取优先级队列信息（如果存在）
                    priority_info = ""
                    if hasattr(scheduler, 'waiting') and scheduler.waiting:
                        # 统计不同优先级的请求
                        priority_counts = {}
                        for seq_group in scheduler.waiting:
                            priority = getattr(seq_group, 'priority', 0)
                            priority_counts[priority] = priority_counts.get(priority, 0) + 1
                        
                        if priority_counts:
                            priority_info = f", 优先级分布: {dict(sorted(priority_counts.items()))}"
                    
                    # 打印队列状态
                    logger.info(f"[vllm engine 队列监控] 等待队列: {waiting_queue_size}, 运行队列: {running_queue_size}, "
                               f"交换队列: {swapped_queue_size}, 总未完成: {total_unfinished}{priority_info}")
                    
                    # 如果有详细的序列信息，打印前几个请求的详情
                    if hasattr(scheduler, 'waiting') and scheduler.waiting and waiting_queue_size > 0:
                        logger.info(f"[vllm engine 队列详情] 等待队列中的前3个请求:")
                        for i, seq_group in enumerate(scheduler.waiting[:3]):
                            request_id = getattr(seq_group, 'request_id', f'seq_{i}')
                            priority = getattr(seq_group, 'priority', 0)
                            arrival_time = getattr(seq_group, 'arrival_time', 0)
                            current_time = time.time()
                            wait_time = current_time - arrival_time if arrival_time > 0 else 0
                            logger.info(f"  [{i+1}] ID: {request_id}, 优先级: {priority}, 等待时间: {wait_time:.2f}s")
                else:
                    logger.warning(f"[vllm engine 队列监控] 调度器不是list或为空: {type(scheduler_list)}")
                
            else:
                logger.warning("[vllm engine 队列监控] 无法访问引擎调度器信息")
                
        except Exception as e:
            logger.error(f"[vllm engine 队列监控] 监控过程中出现错误: {e}")
        
        # 等待指定间隔
        await asyncio.sleep(interval)


def get_gpu_count(logger):
    """获取可用的GPU数量"""
    try:
        # 方法1: 使用nvidia-ml-py (推荐)
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"通过pynvml检测到 {gpu_count} 个GPU")
            return gpu_count
        except ImportError:
            logger.debug("pynvml不可用，使用nvidia-smi")
        
        # 方法2: 使用nvidia-smi --list-gpus (更可靠)
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, check=True)
            gpu_lines = [line for line in result.stdout.strip().split('\n') if line.strip() and 'GPU' in line]
            gpu_count = len(gpu_lines)
            logger.info(f"通过nvidia-smi --list-gpus检测到 {gpu_count} 个GPU")
            return gpu_count
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("nvidia-smi --list-gpus失败，尝试其他方法")
        
        # 方法3: 使用nvidia-smi --query-gpu=count (处理多行输出)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            if lines:
                # 如果有多行，说明有多个GPU，行数就是GPU数量
                gpu_count = len(lines)
                logger.info(f"通过nvidia-smi count查询检测到 {gpu_count} 个GPU")
                return gpu_count
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            logger.debug("nvidia-smi count查询失败")
        
        # 方法4: 使用nvidia-smi -L (备用方法)
        try:
            result = subprocess.run(['nvidia-smi', '-L'], 
                                  capture_output=True, text=True, check=True)
            gpu_lines = [line for line in result.stdout.strip().split('\n') if line.strip() and 'GPU' in line]
            gpu_count = len(gpu_lines)
            logger.info(f"通过nvidia-smi -L检测到 {gpu_count} 个GPU")
            return gpu_count
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("nvidia-smi -L失败")
            
        # 如果所有方法都失败了
        logger.warning("所有GPU检测方法都失败，假设使用CPU")
        return 0
        
    except Exception as e:
        logger.warning(f"GPU检测过程中出现异常: {e}，假设使用CPU")
        return 0


def adjust_engine_config_for_resources(args, logger):
    """根据可用资源调整引擎配置"""
    gpu_count = get_gpu_count(logger)
    
    if gpu_count == 0:
        logger.warning("未检测到GPU，将使用CPU模式（性能会显著降低）")
        args.tensor_parallel_size = 1
        args.gpu_memory_utilization = 0.5
        args.max_num_seqs = 4  # CPU模式下保守一些
    elif gpu_count < args.tensor_parallel_size:
        logger.warning(f"可用GPU数量({gpu_count})少于tensor_parallel_size({args.tensor_parallel_size})，自动调整")
        args.tensor_parallel_size = gpu_count
    
    # 调整max_num_seqs以允许更多并发序列
    # 不要限制为1，这会严重限制并发性能
    if not hasattr(args, 'max_num_seqs') or args.max_num_seqs is None:
        args.max_num_seqs = 64  # 设置合理的默认值
    else:
        # 确保max_num_seqs至少为16，以支持基本的并发处理
        args.max_num_seqs = max(args.max_num_seqs, 16)
    
    # 保守的资源配置
    args.gpu_memory_utilization = min(args.gpu_memory_utilization, 0.8)
    
    logger.info(f"调整后的引擎配置: tensor_parallel_size={args.tensor_parallel_size}, "
                f"gpu_memory_utilization={args.gpu_memory_utilization}, max_num_seqs={args.max_num_seqs}")


async def start_vllm_engine(args, logger):
    """启动vLLM引擎"""
    global queue_monitor_task
    
    if not vllm_available:
        logger.error("vLLM not available, cannot start engine")
        return None
    
    try:
        # 导入配置（仅在需要时导入以避免循环依赖）
        try:
            suppress_logs = GLOBAL_CONFIG.get("suppress_vllm_engine_logs", True)
            log_level = GLOBAL_CONFIG.get("vllm_log_level", "WARNING")
        except ImportError:
            suppress_logs = True  # 默认抑制日志
            log_level = "WARNING"
        
        # 抑制vLLM详细日志
        if suppress_logs:
            setup_vllm_logging(log_level, True)
            logger.info("✓ vLLM详细日志已抑制")
        else:
            logger.info("ⓘ vLLM详细日志保持开启")
        
        # 调整配置以匹配可用资源
        adjust_engine_config_for_resources(args, logger)
        
        # 设置环境变量以减少警告和避免LLVM错误
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
        # 添加LLVM相关环境变量
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        
        # 从args获取引擎参数，如果没有则使用默认值
        engine_args = AsyncEngineArgs(
            model=getattr(args, 'model_path', '/path/to/model'),
            tensor_parallel_size=getattr(args, 'tensor_parallel_size', 8),
            pipeline_parallel_size=getattr(args, 'pipeline_parallel_size', 1),
            gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.9),
            max_model_len=getattr(args, 'max_model_len', 4096),
            max_num_seqs=getattr(args, 'max_num_seqs', 64),
            disable_log_stats=getattr(args, 'disable_log_stats', True),
            enable_prefix_caching=False,  # 强制禁用前缀缓存
            dtype=getattr(args, 'dtype', 'half'),
            quantization=None,  # 暂时禁用量化避免LLVM错误
            scheduling_policy=getattr(args, 'scheduling_policy', 'priority'),
        )
        
        logger.info("Creating AsyncLLMEngine with conservative args:")
        logger.info(f"  model: {engine_args.model}")
        logger.info(f"  tensor_parallel_size: {engine_args.tensor_parallel_size}")
        logger.info(f"  pipeline_parallel_size: {engine_args.pipeline_parallel_size}")
        logger.info(f"  gpu_memory_utilization: {engine_args.gpu_memory_utilization}")
        logger.info(f"  max_model_len: {engine_args.max_model_len}")
        logger.info(f"  max_num_seqs: {engine_args.max_num_seqs}")
        logger.info(f"  quantization: {engine_args.quantization}")
        logger.info(f"  dtype: {engine_args.dtype}")
        logger.info(f"  disable_log_stats: {engine_args.disable_log_stats}")
        logger.info(f"  enable_prefix_caching: {engine_args.enable_prefix_caching}")
        logger.info(f"  scheduling_policy: {engine_args.scheduling_policy}")
        
        # 创建引擎实例
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 测试引擎是否正常工作
        await asyncio.sleep(5)  # 给引擎更多初始化时间
        
        # 启动队列监控任务
        monitor_interval = GLOBAL_CONFIG.get("queue_monitor_interval", 5)
        queue_monitor_task = asyncio.create_task(monitor_engine_queue(engine, monitor_interval, logger))
        logger.info(f"✓ vllm engine 队列监控任务已启动，监控间隔: {monitor_interval}秒")
        
        logger.info("vLLM AsyncLLMEngine started successfully!")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to start vLLM engine: {e}")
        import traceback
        traceback.print_exc()
        return None


def stop_vllm_engine(engine, logger):
    """停止vLLM引擎"""
    global vllm_process, queue_monitor_task
    
    # 停止队列监控任务
    if queue_monitor_task and not queue_monitor_task.done():
        queue_monitor_task.cancel()
        logger.info("队列监控任务已停止")
    
    if engine:
        try:
            # 如果引擎有stop方法，调用它
            if hasattr(engine, 'stop'):
                engine.stop()
            logger.info("vLLM engine stopped")
        except Exception as e:
            logger.warning(f"Error stopping vLLM engine: {e}")
    
    # 清理可能存在的进程
    if vllm_process and vllm_process.poll() is None:
        try:
            # 先尝试优雅关闭
            vllm_process.terminate()
            vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # 如果优雅关闭失败，强制杀死
            vllm_process.kill()
            vllm_process.wait()
        except Exception as e:
            logger.warning(f"Error cleaning up vLLM process: {e}")
        finally:
            vllm_process = None


