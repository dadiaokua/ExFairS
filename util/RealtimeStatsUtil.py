"""
实时监控统计工具模块

包含：
- ClientStats: 客户端统计数据类（窗口模式）
- 公平性计算工具函数
- Jain Index 计算函数
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ClientStats:
    """
    客户端实时统计数据（窗口模式）
    
    设计：
    - 每完成一个请求，实时更新 window_* 字段
    - 每次监控时，读取 window_* 字段的数据
    - 监控完成后，重置 window_* 字段，开始下一个窗口
    - cumulative_* 字段保存全局累计数据（用于最终报告）
    """
    client_id: str
    
    # ===== 当前窗口统计（每次监控后重置）=====
    window_requests: int = 0           # 窗口内请求数
    window_completed: int = 0          # 窗口内成功完成数
    window_slo_violations: int = 0     # 窗口内 SLO 违约数
    window_input_tokens: int = 0       # 窗口内输入 token 数
    window_output_tokens: int = 0      # 窗口内输出 token 数
    window_total_latency_ms: float = 0.0   # 窗口内总延迟 (ms)
    window_total_ttft_ms: float = 0.0      # 窗口内总 TTFT (ms)
    window_total_queue_ms: float = 0.0     # 窗口内总排队时间 (ms)
    window_total_inference_ms: float = 0.0 # 窗口内总推理时间 (ms)
    window_start_time: float = field(default_factory=time.time)
    
    # ===== 累计统计（全局，不重置）=====
    cumulative_requests: int = 0
    cumulative_completed: int = 0
    cumulative_slo_violations: int = 0
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    cumulative_latency_ms: float = 0.0
    cumulative_ttft_ms: float = 0.0
    cumulative_queue_ms: float = 0.0
    cumulative_inference_ms: float = 0.0
    
    # P95/P99 延迟追踪（限制大小以避免内存泄漏）
    latency_samples: List[float] = field(default_factory=list)
    max_latency_samples: int = 10000  # 最大保留样本数
    
    last_update_time: float = field(default_factory=time.time)
    
    def add_latency_sample(self, latency_ms: float):
        """添加延迟样本（带大小限制）"""
        self.latency_samples.append(latency_ms)
        # 如果超过最大样本数，移除最早的样本
        if len(self.latency_samples) > self.max_latency_samples:
            # 保留最近的一半样本，避免频繁删除
            self.latency_samples = self.latency_samples[-self.max_latency_samples // 2:]
    
    # ===== 窗口内实时计算的 metrics =====
    @property
    def slo_violation_rate(self) -> float:
        """当前窗口 SLO 违约率"""
        if self.window_completed == 0:
            return 0.0
        return self.window_slo_violations / self.window_completed
    
    @property
    def avg_latency_ms(self) -> float:
        """当前窗口平均延迟 (ms)"""
        if self.window_completed == 0:
            return 0.0
        return self.window_total_latency_ms / self.window_completed
    
    @property
    def avg_ttft_ms(self) -> float:
        """当前窗口平均 TTFT (ms)"""
        if self.window_completed == 0:
            return 0.0
        return self.window_total_ttft_ms / self.window_completed
    
    @property
    def avg_queue_ms(self) -> float:
        """当前窗口平均排队时间 (ms)"""
        if self.window_completed == 0:
            return 0.0
        return self.window_total_queue_ms / self.window_completed
    
    @property
    def avg_inference_ms(self) -> float:
        """当前窗口平均推理时间 (ms)"""
        if self.window_completed == 0:
            return 0.0
        return self.window_total_inference_ms / self.window_completed
    
    @property
    def tokens_per_second(self) -> float:
        """当前窗口 TPS（输出 token / 总推理时间）"""
        if self.window_total_inference_ms == 0:
            return 0.0
        return self.window_output_tokens / (self.window_total_inference_ms / 1000)
    
    @property
    def service_value(self) -> float:
        """当前窗口服务价值：input_tokens + 2 * output_tokens"""
        return self.window_input_tokens + 2 * self.window_output_tokens
    
    @property
    def completion_rate(self) -> float:
        """当前窗口完成率"""
        if self.window_requests == 0:
            return 0.0
        return self.window_completed / self.window_requests
    
    # ===== 累计统计的 metrics =====
    @property
    def cumulative_slo_violation_rate(self) -> float:
        """累计 SLO 违约率"""
        if self.cumulative_completed == 0:
            return 0.0
        return self.cumulative_slo_violations / self.cumulative_completed
    
    @property
    def cumulative_avg_latency_ms(self) -> float:
        """累计平均延迟 (ms)"""
        if self.cumulative_completed == 0:
            return 0.0
        return self.cumulative_latency_ms / self.cumulative_completed
    
    @property
    def cumulative_avg_queue_ms(self) -> float:
        """累计平均排队时间 (ms)"""
        if self.cumulative_completed == 0:
            return 0.0
        return self.cumulative_queue_ms / self.cumulative_completed
    
    @property
    def cumulative_avg_inference_ms(self) -> float:
        """累计平均推理时间 (ms)"""
        if self.cumulative_completed == 0:
            return 0.0
        return self.cumulative_inference_ms / self.cumulative_completed
    
    @property
    def p95_latency_ms(self) -> float:
        """P95 延迟"""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    @property
    def p99_latency_ms(self) -> float:
        """P99 延迟"""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    # ===== 兼容旧属性名 =====
    @property
    def total_requests(self) -> int:
        return self.window_requests
    
    @property
    def successful_requests(self) -> int:
        return self.window_completed
    
    @property
    def slo_violations(self) -> int:
        return self.window_slo_violations
    
    @property
    def avg_latency(self) -> float:
        return self.avg_latency_ms
    
    @property
    def total_input_tokens(self) -> int:
        return self.window_input_tokens
    
    @property
    def total_output_tokens(self) -> int:
        return self.window_output_tokens
    
    def reset_window(self):
        """重置窗口统计（监控完成后调用）"""
        self.window_requests = 0
        self.window_completed = 0
        self.window_slo_violations = 0
        self.window_input_tokens = 0
        self.window_output_tokens = 0
        self.window_total_latency_ms = 0.0
        self.window_total_ttft_ms = 0.0
        self.window_total_queue_ms = 0.0
        self.window_total_inference_ms = 0.0
        self.window_start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于保存）"""
        return {
            "client_id": self.client_id,
            "window": {
                "requests": self.window_requests,
                "completed": self.window_completed,
                "slo_violations": self.window_slo_violations,
                "slo_violation_rate": self.slo_violation_rate,
                "avg_latency_ms": self.avg_latency_ms,
                "avg_ttft_ms": self.avg_ttft_ms,
                "avg_queue_ms": self.avg_queue_ms,
                "avg_inference_ms": self.avg_inference_ms,
                "tokens_per_second": self.tokens_per_second,
                "completion_rate": self.completion_rate,
            },
            "cumulative": {
                "requests": self.cumulative_requests,
                "completed": self.cumulative_completed,
                "slo_violations": self.cumulative_slo_violations,
                "slo_violation_rate": self.cumulative_slo_violation_rate,
                "avg_latency_ms": self.cumulative_avg_latency_ms,
                "avg_queue_ms": self.cumulative_avg_queue_ms,
                "avg_inference_ms": self.cumulative_avg_inference_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "p99_latency_ms": self.p99_latency_ms,
                "input_tokens": self.cumulative_input_tokens,
                "output_tokens": self.cumulative_output_tokens,
            }
        }


def calculate_jain_index(values: List[float]) -> float:
    """
    计算 Jain's Fairness Index
    
    对于 SLO 违约率（越小越好），使用 1 - normalized_value 转换
    
    Args:
        values: 各客户端的指标值列表
        
    Returns:
        Jain's Fairness Index (0-1)
    """
    n = len(values)
    
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return 1.0  # 所有值相等，完全公平
    
    # 归一化并转换（越小越好 -> 1 - normalized）
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    transformed = [1 - nv for nv in normalized]
    
    # Jain's Index
    sum_t = sum(transformed)
    sum_sq = sum(t**2 for t in transformed)
    
    if n * sum_sq == 0:
        return 0.0
    
    return (sum_t ** 2) / (n * sum_sq)


def calculate_fairness_metrics(all_stats: Dict[str, ClientStats]) -> Dict[str, Any]:
    """
    计算公平性指标
    
    Args:
        all_stats: 所有客户端的统计数据
        
    Returns:
        公平性计算结果
    """
    # 计算总服务价值
    total_service = sum(s.service_value for s in all_stats.values())
    
    if total_service == 0:
        return {
            "jain_index": 1.0,
            "total_service": 0,
            "slo_violation_rates": {}
        }
    
    # 计算各客户端的 SLO 违约率
    slo_violation_rates = {
        cid: stats.slo_violation_rate
        for cid, stats in all_stats.items()
    }
    
    # 计算 Jain's Fairness Index（基于 SLO 违约率）
    jain_index = calculate_jain_index(list(slo_violation_rates.values()))
    
    return {
        "jain_index": jain_index,
        "total_service": total_service,
        "slo_violation_rates": slo_violation_rates
    }


def calculate_alpha_adjustment(all_stats: Dict[str, ClientStats], 
                               current_alpha: float,
                               alpha_min: float = 0.3,
                               alpha_max: float = 0.95,
                               adjust_rate: float = 0.1) -> float:
    """
    动态计算 alpha 调整值
    
    策略：
    - 如果 SLO 违约率方差大（不均衡），增大 alpha，让调度更关注 SLO
    - 如果 SLO 违约率已经很均衡，减小 alpha，让调度平衡考虑服务量
    
    Args:
        all_stats: 所有客户端统计数据
        current_alpha: 当前 alpha 值
        alpha_min: alpha 最小值
        alpha_max: alpha 最大值
        adjust_rate: 调整幅度
        
    Returns:
        新的 alpha 值
    """
    if len(all_stats) < 2:
        return current_alpha
    
    # 计算 SLO 违约率的标准差
    slo_rates = [s.slo_violation_rate for s in all_stats.values()]
    mean_slo = sum(slo_rates) / len(slo_rates)
    variance = sum((r - mean_slo) ** 2 for r in slo_rates) / len(slo_rates)
    std_slo = variance ** 0.5
    
    new_alpha = current_alpha
    
    # 根据 SLO 违约率的不均衡程度调整 alpha
    if std_slo > 0.2:
        # 不均衡严重，增大 alpha
        new_alpha = min(alpha_max, current_alpha + adjust_rate)
    elif std_slo < 0.05:
        # 已经均衡，适当减小 alpha
        new_alpha = max(alpha_min, current_alpha - adjust_rate * 0.5)
    
    return new_alpha


def format_monitor_summary(monitor_count: int, 
                          all_stats: Dict[str, ClientStats],
                          jain_index: float,
                          alpha: float,
                          exchange_count: int,
                          client_priorities: Dict[str, int]) -> str:
    """
    格式化监控摘要输出
    
    Args:
        monitor_count: 监控次数
        all_stats: 所有客户端统计
        jain_index: Jain Index
        alpha: 当前 alpha 值
        exchange_count: 优先级交换次数
        client_priorities: 客户端优先级字典
        
    Returns:
        格式化的字符串
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Monitor #{monitor_count} Summary (Window Data)")
    lines.append(f"{'='*80}")
    lines.append(f"Jain's Index: {jain_index:.4f} | Alpha: {alpha:.3f} | Priority Exchanges: {exchange_count}")
    lines.append("")
    
    # 表头
    header = (f"{'Client':<12} {'Req':<6} {'Done':<6} {'SLO%':<8} "
             f"{'Lat(ms)':<10} {'TTFT':<8} {'TPS':<8} {'Priority':<8}")
    lines.append(header)
    lines.append("-" * 80)
    
    # 每个客户端的窗口数据
    total_requests = 0
    total_completed = 0
    total_violations = 0
    
    for cid, stats in all_stats.items():
        priority = client_priorities.get(cid, 0)
        row = (f"{cid:<12} "
              f"{stats.window_requests:<6} "
              f"{stats.window_completed:<6} "
              f"{stats.slo_violation_rate*100:<8.1f} "
              f"{stats.avg_latency_ms:<10.1f} "
              f"{stats.avg_ttft_ms:<8.1f} "
              f"{stats.tokens_per_second:<8.1f} "
              f"{priority:<8}")
        lines.append(row)
        
        total_requests += stats.window_requests
        total_completed += stats.window_completed
        total_violations += stats.window_slo_violations
    
    # 汇总
    lines.append("-" * 80)
    overall_slo_rate = (total_violations / total_completed * 100) if total_completed > 0 else 0
    lines.append(f"{'TOTAL':<12} {total_requests:<6} {total_completed:<6} {overall_slo_rate:<8.1f}")
    lines.append(f"{'='*80}\n")
    
    return "\n".join(lines)
