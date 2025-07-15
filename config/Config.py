from datetime import datetime

def get_monitor_file_time():
    """动态生成监控文件时间戳"""
    return datetime.now().strftime("%m_%d_%H_%M")

GLOBAL_CONFIG = {
    "latency_slo": 5,
    "output_tokens": 256,
    "alpha": 0.7,
    "fairness_ratio_LFS": 0.1,
    "fairness_ratio_VTC": 0.5,
    'ADJUST_SENSITIVITY': 1,
    "b": 1.5,
    "whether_fairness": 1,
    "max_granularity": 10,
    "round_time": 60,
    "monitor_file_time": get_monitor_file_time(),  # 动态生成时间戳
    "exp_time": 36000,
    "avg_success_rate": 0.9,
    "max_exchange_times": 1,
    "prompt_max_len": 10000,
    "request_model_name": "",
    "buffer_ratio": 0.2,
    
    # AsyncLLMEngine采样参数
    "sampling_temperature": 0.7,
    "sampling_top_p": 0.9,
    "sampling_top_k": -1,  # -1表示不使用top_k
    "sampling_repetition_penalty": 1.0,
    
    # vLLM日志控制
    "vllm_log_level": "WARNING",  # 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "suppress_vllm_engine_logs": True,  # 是否抑制引擎请求/完成日志
    
    # 队列监控配置
    "queue_monitor_interval": 5,  # 队列监控间隔（秒）
    "queue_worker_sleep_time": 1,  # 队列列表睡眠时间（秒）
}
