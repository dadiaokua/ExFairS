from datetime import datetime

def get_monitor_file_time():
    """动态生成监控文件时间戳"""
    return datetime.now().strftime("%m_%d_%H_%M")

GLOBAL_CONFIG = {
    "latency_slo": 5,
    "output_tokens": 512,
    "alpha": 0.8,                       # SLO 违约率在公平性计算中的权重（静态时使用）
    "fairness_ratio_exfairs": 0.05,    # ExFairS 触发阈值（差值>此值才调整）
    
    # 动态 α 配置（焦虑驱动型）
    "dynamic_alpha_enabled": True,      # 是否启用动态 α
    "alpha_min": 0.2,                   # 最小 α（关注资源公平）
    "alpha_max": 0.9,                   # 最大 α（关注 SLO）
    "alpha_k": 10.0,                    # Sigmoid 陡峭度
    "alpha_theta": 0.1,                 # SLO 违约率警戒线（10%）
    "fairness_ratio_VTC": 0.5,         # VTC 策略触发阈值
    'ADJUST_SENSITIVITY': 1.5,         # 优先级调整灵敏度（越大调整幅度越大）
    "whether_fairness": 1,
    "max_granularity": 10,
    "round_time": 60,
    "monitor_file_time": get_monitor_file_time(),  # 动态生成时间戳
    "exp_time": 36000,
    "avg_success_rate": 0.9,
    "max_exchange_times": 1,
    "prompt_max_len": 10000,
    "request_model_name": "Qwen/Qwen3-8B",
    "buffer_ratio": 0.05,  # 缓冲时间比例，用于等待最后的请求完成
    "request_timeout": 30,  # 单个请求超时时间（秒），会被场景配置覆盖
    
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

    # QuE
    "que_throughput": 0.3,
    "que_latency": 0.4,
    "que_cost": 0.3,
    
    # ExFairS 优先级控制参数
    "priority_min": -50,              # 优先级下界（最高优先级）
    "priority_max": 50,               # 优先级上界（最低优先级）
    "priority_decay_rate": 0.95,      # 每轮优先级衰减系数（向0收敛），0.95比0.9衰减更慢
    "priority_amplifier": 15,         # 优先级放大系数（从10提升到15，让调整更显著）
    "fairness_window_size": 1,        # 时间窗大小（1=只使用前一轮数据计算公平性）
    
    # 实时监控模式配置
    "realtime_mode": False,           # 是否启用实时监控模式
    "realtime_duration": 600,         # 实时模式总持续时间（秒）
    "realtime_monitor_interval": 60,  # 实时监控间隔（秒）
}
