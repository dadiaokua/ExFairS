#!/usr/bin/env python3
"""
参数解析模块
处理命令行参数的解析、验证和预处理
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def safe_float_conversion(value, default=0):
    """安全地将值转换为浮点数"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_args(logger):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='vLLM Benchmark Tool')
    
    # 基础连接参数
    parser.add_argument('--vllm_url', type=str, default="http://localhost:8000/v1", help='vLLM server URL')
    parser.add_argument('--api_key', type=str, default="test", help='API key for vLLM server')
    parser.add_argument('--use_tunnel', type=int, default=0, help='Use SSH tunnel (0 or 1)')
    parser.add_argument('--local_port', type=str, default="8000", help='Local port(s), can be comma-separated')
    parser.add_argument('--remote_port', type=str, default="8000", help='Remote port(s), can be comma-separated')
    
    # 请求配置参数
    parser.add_argument('--distribution', type=str, default="poisson", help='Request distribution type')
    parser.add_argument('--short_qpm', type=str, default="60", help='Short request QPM, space-separated')
    parser.add_argument('--short_client_qpm_ratio', type=float, default=1.0, help='Short client QPM ratio')
    parser.add_argument('--long_qpm', type=str, default="60", help='Long request QPM, space-separated')
    parser.add_argument('--long_client_qpm_ratio', type=float, default=1.0, help='Long client QPM ratio')
    parser.add_argument('--mix_qpm', type=str, default="60", help='MIX request QPM, space-separated')
    parser.add_argument('--mix_client_qpm_ratio', type=float, default=1.0, help='MIX client QPM ratio')
    
    # 客户端配置参数
    parser.add_argument('--short_clients', type=int, default=1, help='Number of short request clients')
    parser.add_argument('--short_clients_slo', type=str, default="10", help='Short client SLO, space-separated')
    parser.add_argument('--long_clients', type=int, default=1, help='Number of long request clients')
    parser.add_argument('--long_clients_slo', type=str, default="10", help='Long client SLO, space-separated')
    parser.add_argument('--mix_clients', type=int, default=0, help='Number of MIX request clients')
    parser.add_argument('--mix_clients_slo', type=str, default="10", help='MIX client SLO, space-separated')
    
    # 并发和性能参数
    parser.add_argument('--concurrency', type=int, default=5, help='Number of concurrent requests')
    parser.add_argument('--num_requests', type=int, default=100, help='Total number of requests')
    parser.add_argument('--request_timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--sleep', type=int, default=1, help='Sleep time between rounds')
    
    # 实验配置参数
    parser.add_argument('--round', type=int, default=5, help='Number of benchmark rounds')
    parser.add_argument('--round_time', type=int, default=300, help='Time limit per round in seconds')
    parser.add_argument('--exp', type=str, default="LFS", help='Experiment type')
    parser.add_argument('--use_time_data', type=int, default=0, help='Use time data (0 or 1)')
    
    # 模型和tokenizer参数
    parser.add_argument('--tokenizer', type=str, 
                       default="/home/llm/model_hub/Qwen2.5-32B-Instruct", 
                       help='Path to tokenizer')
    parser.add_argument('--request_model_name', type=str, 
                       default="Qwen2.5-32B", 
                       help='Model name for requests')
    
    # vLLM引擎参数（新增）
    parser.add_argument('--start_engine', type=bool, default=True, help='Whether to start vLLM engine')
    parser.add_argument('--model_path', type=str, 
                       default="/home/llm/model_hub/Qwen2.5-32B-Instruct", 
                       help='Path to model')
    parser.add_argument('--tensor_parallel_size', type=int, default=8, help='Tensor parallel size')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
    parser.add_argument('--max_model_len', type=int, default=8124, help='Maximum model length')
    parser.add_argument('--trust_remote_code', type=bool, default=True, help='Trust remote code')
    parser.add_argument('--disable_log_stats', type=bool, default=True, help='Disable log stats')
    parser.add_argument('--enable_prefix_caching', type=bool, default=False, help='Enable prefix caching')
    parser.add_argument('--swap_space', type=int, default=0, help='Swap space in GB')
    parser.add_argument('--dtype', type=str, default="float16", help='Data type')
    parser.add_argument('--quantization', type=str, default="None", help='Quantization method')
    parser.add_argument('--max_num_seqs', type=int, default=128, help='Maximum number of sequences') 
    parser.add_argument('--max_num_batched_tokens', type=int, default=65536, help='Maximum number of batched tokens')
    parser.add_argument('--scheduling_policy', type=str, default="priority", help='Scheduling policy')
    
    args = parser.parse_args()
    return args


def validate_args(args, logger):
    """验证和预处理参数"""
    try:
        # 处理端口参数
        if isinstance(args.local_port, str):
            if ',' in args.local_port:
                args.local_port = [int(p.strip()) for p in args.local_port.split(',')]
            else:
                args.local_port = int(args.local_port)
        
        if isinstance(args.remote_port, str):
            if ',' in args.remote_port:
                args.remote_port = [int(p.strip()) for p in args.remote_port.split(',')]
            else:
                args.remote_port = int(args.remote_port)
        
        # 处理QPM参数
        if isinstance(args.short_qpm, str):
            args.short_qpm = [safe_float_conversion(q.strip()) for q in args.short_qpm.split()]
        
        if isinstance(args.long_qpm, str):
            args.long_qpm = [safe_float_conversion(q.strip()) for q in args.long_qpm.split()]
        
        if isinstance(args.mix_qpm, str):
            args.mix_qpm = [safe_float_conversion(q.strip()) for q in args.mix_qpm.split()]
        
        # 处理SLO参数
        if isinstance(args.short_clients_slo, str):
            args.short_clients_slo = [safe_float_conversion(s.strip(), 10) for s in args.short_clients_slo.split()]
        
        if isinstance(args.long_clients_slo, str):
            args.long_clients_slo = [safe_float_conversion(s.strip(), 10) for s in args.long_clients_slo.split()]
        
        if isinstance(args.mix_clients_slo, str):
            args.mix_clients_slo = [safe_float_conversion(s.strip(), 10) for s in args.mix_clients_slo.split()]
        
        # 验证参数数量匹配
        if len(args.short_qpm) > 1 and len(args.short_qpm) != args.short_clients:
            logger.warning(f"Short QPM count ({len(args.short_qpm)}) doesn't match short clients ({args.short_clients})")
        
        if len(args.long_qpm) > 1 and len(args.long_qpm) != args.long_clients:
            logger.warning(f"Long QPM count ({len(args.long_qpm)}) doesn't match long clients ({args.long_clients})")
        
        if len(args.mix_qpm) > 1 and len(args.mix_qpm) != args.mix_clients:
            logger.warning(f"MIX QPM count ({len(args.mix_qpm)}) doesn't match MIX clients ({args.mix_clients})")
        
        if len(args.short_clients_slo) > 1 and len(args.short_clients_slo) != args.short_clients:
            logger.warning(f"Short client SLO count ({len(args.short_clients_slo)}) doesn't match short clients ({args.short_clients})")
        
        if len(args.long_clients_slo) > 1 and len(args.long_clients_slo) != args.long_clients:
            logger.warning(f"Long client SLO count ({len(args.long_clients_slo)}) doesn't match long clients ({args.long_clients})")
        
        if len(args.mix_clients_slo) > 1 and len(args.mix_clients_slo) != args.mix_clients:
            logger.warning(f"MIX client SLO count ({len(args.mix_clients_slo)}) doesn't match MIX clients ({args.mix_clients})")
        
        return args
        
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        return None


def print_benchmark_config(args, logger):
    """打印基准测试配置"""
    logger.info("=== Benchmark Configuration ===")
    logger.info(f"vLLM URL: {args.vllm_url}")
    logger.info(f"API Key: {args.api_key}")
    logger.info(f"Use Tunnel: {args.use_tunnel}")
    logger.info(f"Local Port: {args.local_port}")
    logger.info(f"Remote Port: {args.remote_port}")
    logger.info(f"Distribution: {args.distribution}")
    logger.info(f"Short QPM: {args.short_qpm}")
    logger.info(f"Long QPM: {args.long_qpm}")
    logger.info(f"MIX QPM: {args.mix_qpm}")
    logger.info(f"Short Clients: {args.short_clients}")
    logger.info(f"Long Clients: {args.long_clients}")
    logger.info(f"Mix Clients: {args.mix_clients}")
    logger.info(f"Short Client SLO: {args.short_clients_slo}")
    logger.info(f"Long Client SLO: {args.long_clients_slo}")
    logger.info(f"MIX Client SLO: {args.mix_clients_slo}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Num Requests: {args.num_requests}")
    logger.info(f"Request Timeout: {args.request_timeout}")
    logger.info(f"Rounds: {args.round}")
    logger.info(f"Round Time: {args.round_time}")
    logger.info(f"Experiment: {args.exp}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Request Model Name: {args.request_model_name}")
    
    # 如果启动引擎，显示引擎配置
    if getattr(args, 'start_engine', True):
        logger.info("=== vLLM Engine Configuration ===")
        logger.info(f"Model Path: {args.model_path}")
        logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")
        logger.info(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
        logger.info(f"Max Model Length: {args.max_model_len}")
        logger.info(f"Max Num Seqs: {args.max_num_seqs}")
        logger.info(f"Trust Remote Code: {args.trust_remote_code}")
        logger.info(f"Disable Log Stats: {args.disable_log_stats}")
        logger.info(f"Enable Prefix Caching: {args.enable_prefix_caching}")
        logger.info(f"Swap Space: {args.swap_space}")
        logger.info(f"Data Type: {args.dtype}")
        logger.info(f"Quantization: {args.quantization}")
    
    logger.info("===============================")


def parse_and_validate_arguments():
    """解析和验证命令行参数的统一函数"""
    # 解析参数
    args = parse_args(logger)
    
    # 打印配置
    print_benchmark_config(args, logger)
    
    # 验证参数
    validated_args = validate_args(args, logger)
    
    return validated_args 