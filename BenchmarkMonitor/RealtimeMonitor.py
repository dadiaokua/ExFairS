"""
实时监控系统 - RealtimeMonitor

核心设计：
1. 只做一轮，持续时间长（如10分钟）
2. 后台监控线程每60秒收集一次数据
3. 实时计算SAFI和优先级交换
4. 不阻塞推理，更新后的优先级立即作用于新请求

时间线:
T0 -------- T60s -------- T120s -------- T180s -------- ... -------- T600s (10min)
   |  窗口1  |   窗口2    |   窗口3     |   窗口4                    |
   | 监控&重置|  监控&重置 |  监控&重置  |  监控&重置                |  结束
"""

import asyncio
import logging
import os
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional

from config.Config import GLOBAL_CONFIG
from util.RealtimeStatsUtil import (
    ClientStats, 
    calculate_jain_index,
    calculate_fairness_metrics,
    calculate_alpha_adjustment,
    format_monitor_summary
)


class RealtimeMonitor:
    """
    实时监控器
    
    功能：
    1. 后台线程每60秒收集一次所有客户端数据
    2. 实时计算SAFI和fairness_ratio
    3. 进行优先级交换
    4. 更新后的优先级立即作用于新请求
    """
    
    def __init__(self, 
                 clients: List, 
                 exp_type: str,
                 monitor_interval: int = 60,
                 total_duration: int = 600,
                 config: Dict = None):
        """
        初始化实时监控器
        
        Args:
            clients: 客户端列表
            exp_type: 实验类型
            monitor_interval: 监控间隔（秒），默认60秒
            total_duration: 总持续时间（秒），默认600秒（10分钟）
            config: 配置参数
        """
        self.clients = clients
        self.exp_type = exp_type
        self.monitor_interval = monitor_interval
        self.total_duration = total_duration
        self.config = config or GLOBAL_CONFIG
        
        # 状态管理
        self.is_running = False
        self.start_time: Optional[float] = None
        self.monitor_task: Optional[asyncio.Task] = None
        
        # 实时统计数据（线程安全）
        self._stats_lock = threading.Lock()
        self._client_stats: Dict[str, ClientStats] = {}
        
        # 监控历史记录
        self.monitor_history: List[Dict] = []
        
        # Alpha 动态调整参数
        self.alpha = self.config.get("alpha", 0.8)
        self.alpha_min = 0.3
        self.alpha_max = 0.95
        self.alpha_adjust_rate = 0.1
        
        # 优先级更新锁（保护优先级交换的原子性）
        self._priority_lock = threading.Lock()
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 初始化客户端统计
        for client in clients:
            self._client_stats[client.client_id] = ClientStats(client_id=client.client_id)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器
        
        控制台只显示重要信息（WARNING及以上），详细信息写入日志文件
        """
        logger = logging.getLogger(f"RealtimeMonitor-{self.exp_type}")
        
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            
            # 控制台处理器 - 只显示警告和错误
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            
            os.makedirs('log', exist_ok=True)
            timestamp = GLOBAL_CONFIG.get("monitor_file_time", datetime.now().strftime("%m%d_%H%M"))
            # 文件处理器 - 记录所有详细信息
            fh = logging.FileHandler(
                filename=f'log/realtime_monitor_{self.exp_type}_{timestamp}.log',
                encoding="utf-8", mode="a"
            )
            fh.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            
            logger.addHandler(ch)
            logger.addHandler(fh)
            logger.propagate = False
        
        return logger
    
    def update_client_stats(self, client_id: str, result: tuple):
        """
        更新客户端统计数据（由请求完成时调用，线程安全）
        
        Args:
            client_id: 客户端ID
            result: 请求结果元组 (tokens, elapsed_time_sec, tps, ttft_ms, input_token, slo, queue_wait_sec)
        """
        with self._stats_lock:
            if client_id not in self._client_stats:
                self._client_stats[client_id] = ClientStats(client_id=client_id)
            
            stats = self._client_stats[client_id]
            
            # 更新请求计数
            stats.window_requests += 1
            stats.cumulative_requests += 1
            
            if result is None or len(result) < 6:
                return
            
            tokens, elapsed_time_sec, tps, ttft_ms, input_token, slo = result[:6]
            queue_wait_sec = result[6] if len(result) > 6 else 0
            
            # 请求成功完成
            if tokens is not None and elapsed_time_sec is not None:
                latency_ms = elapsed_time_sec * 1000
                ttft_value = ttft_ms if ttft_ms is not None else 0
                queue_ms = queue_wait_sec * 1000 if queue_wait_sec else ttft_value
                inference_ms = max(0, latency_ms - queue_ms)
                
                # 更新窗口统计
                stats.window_completed += 1
                stats.window_output_tokens += tokens
                stats.window_total_latency_ms += latency_ms
                stats.window_total_ttft_ms += ttft_value
                stats.window_total_queue_ms += queue_ms
                stats.window_total_inference_ms += inference_ms
                
                # 更新累计统计
                stats.cumulative_completed += 1
                stats.cumulative_latency_ms += latency_ms
                stats.cumulative_ttft_ms += ttft_value
                stats.cumulative_queue_ms += queue_ms
                stats.cumulative_inference_ms += inference_ms
                
                # 记录延迟样本（用于 P95/P99，带大小限制）
                stats.add_latency_sample(latency_ms)
                
            if input_token is not None:
                stats.window_input_tokens += input_token
                stats.cumulative_input_tokens += input_token
                
            if tokens is not None:
                stats.cumulative_output_tokens += tokens
                
            # SLO 违约检查
            if slo == 0:
                stats.window_slo_violations += 1
                stats.cumulative_slo_violations += 1
            
            stats.last_update_time = time.time()
    
    def get_all_stats(self) -> Dict[str, ClientStats]:
        """获取所有客户端统计数据（线程安全）"""
        with self._stats_lock:
            return dict(self._client_stats)
    
    def get_client_stats(self, client_id: str) -> Optional[ClientStats]:
        """获取单个客户端统计数据"""
        with self._stats_lock:
            return self._client_stats.get(client_id)
    
    async def start(self):
        """启动实时监控"""
        if self.is_running:
            self.logger.warning("Realtime monitor is already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info(f"="*60)
        self.logger.info(f"Starting realtime monitor for {self.exp_type}")
        self.logger.info(f"Monitor interval: {self.monitor_interval}s")
        self.logger.info(f"Total duration: {self.total_duration}s ({self.total_duration/60:.1f} min)")
        self.logger.info(f"Number of clients: {len(self.clients)}")
        self.logger.info(f"="*60)
        
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """停止实时监控"""
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Realtime monitor stopped")
    
    async def _monitor_loop(self):
        """监控主循环"""
        monitor_count = 0
        
        while self.is_running:
            elapsed = time.time() - self.start_time
            
            if elapsed >= self.total_duration:
                self.logger.info(f"Total duration reached ({self.total_duration}s), stopping monitor")
                break
            
            # 计算下次监控的等待时间
            # 第一次监控在 monitor_interval 后，之后每隔 monitor_interval 执行
            next_monitor_time = (monitor_count + 1) * self.monitor_interval
            wait_time = max(0, next_monitor_time - elapsed)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            if not self.is_running:
                break
            
            # 检查是否超过总时间
            elapsed = time.time() - self.start_time
            if elapsed >= self.total_duration:
                self.logger.info(f"Total duration reached during wait ({self.total_duration}s), stopping monitor")
                break
            
            monitor_count += 1
            await self._perform_monitoring(monitor_count)
        
        # 执行最终监控（确保最后一个窗口的数据被收集，即使没有常规监控点）
        self.logger.info(f"Executing final monitoring after {monitor_count} regular points")
        await self._perform_monitoring(monitor_count + 1, is_final=True)
        
        self.is_running = False
        self.logger.info(f"Monitor loop completed after {monitor_count} monitoring points (+ 1 final)")
    
    async def _perform_monitoring(self, monitor_count: int, is_final: bool = False):
        """执行一次监控
        
        Args:
            monitor_count: 监控次数
            is_final: 是否为最终监控（实验结束时的收尾监控）
        """
        try:
            # 使用锁保护整个监控-重置过程，防止数据丢失
            with self._stats_lock:
                # 获取统计数据的快照
                all_stats_snapshot = {k: self._copy_stats(v) for k, v in self._client_stats.items()}
                
                # 计算窗口持续时间
                if all_stats_snapshot:
                    min_start_time = min(
                        (s.window_start_time for s in all_stats_snapshot.values()),
                        default=time.time()
                    )
                    window_duration = time.time() - min_start_time
                else:
                    window_duration = 0.0
                
                # 在持有锁的情况下重置窗口统计，确保不会丢失数据
                if not is_final:
                    for stats in self._client_stats.values():
                        stats.reset_window()
            
            # 以下操作在锁外执行，使用快照数据
            
            # 计算公平性指标
            fairness_result = calculate_fairness_metrics(all_stats_snapshot)
            
            # 动态更新 alpha
            old_alpha = self.alpha
            self.alpha = calculate_alpha_adjustment(
                all_stats_snapshot, self.alpha, 
                self.alpha_min, self.alpha_max, self.alpha_adjust_rate
            )
            if abs(old_alpha - self.alpha) > 0.001:
                self.logger.info(f"[Alpha Update] alpha: {old_alpha:.3f} -> {self.alpha:.3f}")
            
            # 更新客户端的 fairness_ratio
            self._update_client_fairness_ratios(all_stats_snapshot, fairness_result)
            
            # 执行优先级调整（使用优先级锁）
            exchange_count = 0
            if "LFS" in self.exp_type or "ExFairS" in self.exp_type:
                exchange_count = self._adjust_priorities(all_stats_snapshot)
                self._reset_exchange_times()
            
            # 记录监控结果
            monitor_record = self._build_monitor_record(
                monitor_count, window_duration, all_stats_snapshot, 
                fairness_result, exchange_count
            )
            self.monitor_history.append(monitor_record)
            
            # 打印监控摘要
            with self._priority_lock:
                client_priorities = {c.client_id: c.priority for c in self.clients}
            summary = format_monitor_summary(
                monitor_count, all_stats_snapshot, 
                fairness_result['jain_index'], self.alpha,
                exchange_count, client_priorities
            )
            log_prefix = "[FINAL] " if is_final else ""
            self.logger.info(f"{log_prefix}{summary}")
            
        except Exception as e:
            self.logger.error(f"Error in monitoring: {e}", exc_info=True)
    
    def _copy_stats(self, stats: ClientStats) -> ClientStats:
        """创建 ClientStats 的深拷贝（用于快照）"""
        from copy import deepcopy
        return deepcopy(stats)
    
    def _build_monitor_record(self, monitor_count: int, window_duration: float,
                             all_stats: Dict[str, ClientStats],
                             fairness_result: Dict, exchange_count: int) -> Dict:
        """构建监控记录"""
        return {
            "monitor_count": monitor_count,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "window_duration_sec": window_duration,
            "jain_index": fairness_result.get("jain_index", 0),
            "exchange_count": exchange_count,
            "alpha": self.alpha,
            "client_stats": {
                cid: {
                    "window_requests": s.window_requests,
                    "window_completed": s.window_completed,
                    "window_slo_violations": s.window_slo_violations,
                    "slo_violation_rate": s.slo_violation_rate,
                    "avg_latency_ms": s.avg_latency_ms,
                    "avg_ttft_ms": s.avg_ttft_ms,
                    "avg_queue_ms": s.avg_queue_ms,
                    "avg_inference_ms": s.avg_inference_ms,
                    "tokens_per_second": s.tokens_per_second,
                    "cumulative_requests": s.cumulative_requests,
                    "cumulative_completed": s.cumulative_completed,
                    "cumulative_slo_violations": s.cumulative_slo_violations,
                    "fairness_ratio": getattr(self._get_client(cid), 'fairness_ratio', 0),
                    "priority": getattr(self._get_client(cid), 'priority', 0)
                }
                for cid, s in all_stats.items()
            }
        }
    
    def _get_client(self, client_id: str):
        """获取客户端对象"""
        for client in self.clients:
            if client.client_id == client_id:
                return client
        return None
    
    def get_client_priority(self, client_id: str) -> int:
        """安全获取客户端优先级（线程安全）"""
        with self._priority_lock:
            for client in self.clients:
                if client.client_id == client_id:
                    return client.priority
        return 0
    
    def _update_client_fairness_ratios(self, all_stats: Dict[str, ClientStats], 
                                       fairness_result: Dict):
        """更新客户端的 fairness_ratio"""
        total_service = fairness_result.get("total_service", 0)
        
        for client in self.clients:
            stats = all_stats.get(client.client_id)
            if stats is None:
                continue
            
            slo_ratio = stats.slo_violation_rate
            service_ratio = stats.service_value / total_service if total_service > 0 else 0
            client.fairness_ratio = self.alpha * slo_ratio + (1 - self.alpha) * service_ratio
    
    def _adjust_priorities(self, all_stats: Dict[str, ClientStats] = None) -> int:
        """执行优先级调整（线程安全）
        
        Args:
            all_stats: 客户端统计数据快照（可选，用于判断SLO违约率）
        """
        if len(self.clients) < 2:
            return 0
        
        exchange_count = 0
        max_iterations = len(self.clients) // 2
        
        # 使用优先级锁保护整个优先级调整过程
        with self._priority_lock:
            for _ in range(max_iterations):
                sorted_clients = sorted(self.clients, key=lambda c: c.fairness_ratio)
                
                threshold = self.config.get('fairness_ratio_exfairs', 0.05)
                max_exchange_times = self.config.get('max_exchange_times', 3)
                
                eligible = [c for c in sorted_clients 
                           if c.exchange_Resources_Times < max_exchange_times]
                
                if len(eligible) < 2:
                    break
                
                low_client = eligible[0]
                high_client = eligible[-1]
                diff = abs(high_client.fairness_ratio - low_client.fairness_ratio)
                
                if diff <= threshold:
                    break
                
                # 执行优先级交换
                sensitivity = self.config.get("ADJUST_SENSITIVITY", 2.0)
                amplifier = self.config.get("priority_amplifier", 10)
                priority_change = max(1, int(diff * sensitivity * amplifier))
                
                # 使用传入的统计数据快照判断SLO违约率
                if all_stats:
                    high_stats = all_stats.get(high_client.client_id)
                    if high_stats and high_stats.slo_violation_rate > 0.5:
                        priority_change = int(priority_change * 1.5)
                
                priority_min = self.config.get("priority_min", -50)
                priority_max = self.config.get("priority_max", 50)
                
                old_high = high_client.priority
                old_low = low_client.priority
                
                high_client.priority = max(priority_min, high_client.priority - priority_change)
                low_client.priority = min(priority_max, low_client.priority + priority_change)
                
                high_client.exchange_Resources_Times += 1
                low_client.exchange_Resources_Times += 1
                exchange_count += 1
                
                self.logger.info(f"[ExFairS] Priority: {high_client.client_id}: {old_high}->{high_client.priority}, "
                               f"{low_client.client_id}: {old_low}->{low_client.priority}")
        
        return exchange_count
    
    def _reset_exchange_times(self):
        """重置交换次数"""
        for client in self.clients:
            client.exchange_Resources_Times = 0
    
    def _reset_all_windows(self):
        """重置所有窗口统计
        
        注意：此方法现在主要由 _perform_monitoring 内部调用，
        在持有 _stats_lock 的情况下直接重置，以避免竞态条件。
        此方法保留用于向后兼容和手动重置场景。
        """
        with self._stats_lock:
            for stats in self._client_stats.values():
                stats.reset_window()
        self.logger.debug("All client windows reset")
    
    def get_final_results(self) -> Dict:
        """获取最终结果（兼容可视化脚本格式）"""
        final_stats = self.get_all_stats()
        
        total_sent = sum(s.cumulative_requests for s in final_stats.values())
        total_completed = sum(s.cumulative_completed for s in final_stats.values())
        total_slo_violations = sum(s.cumulative_slo_violations for s in final_stats.values())
        
        slo_rates = [s.cumulative_slo_violation_rate for s in final_stats.values()]
        final_jain = calculate_jain_index(slo_rates)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "duration": self.total_duration,
            "strategy": self.exp_type.replace("QUEUE_", "").lower(),
            "summary": {
                "total_sent": total_sent,
                "total_completed": total_completed,
                "total_slo_violations": total_slo_violations,
                "total_timeout": total_sent - total_completed,
                "monitor_count": len(self.monitor_history),
                "monitor_interval": self.monitor_interval,
                "final_alpha": self.alpha
            },
            "users": {
                cid: {
                    "stats": {
                        "count": s.cumulative_completed,
                        "avg_total_latency_ms": s.cumulative_avg_latency_ms,
                        "p95_latency_ms": s.p95_latency_ms,
                        "p99_latency_ms": s.p99_latency_ms,
                        "avg_queue_latency_ms": s.cumulative_avg_queue_ms,
                        "avg_inference_latency_ms": s.cumulative_avg_inference_ms,
                        "successful": s.cumulative_completed,
                        "slo_violations": s.cumulative_slo_violations,
                        "timeouts": s.cumulative_requests - s.cumulative_completed
                    }
                }
                for cid, s in final_stats.items()
            },
            "fairness": {
                "jain_index_safi": final_jain,
                "jain_index_slo_violation": final_jain
            },
            "history": self.monitor_history
        }
