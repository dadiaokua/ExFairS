"""
SLO Violation Rate Greedy Scheduling Experiment

基于 SLO 违约率的贪心调度策略。
优先调度 SLO 违约率更高的客户端，以改善整体服务质量。

核心思想：
1. 计算每个客户端的 SLO 违约率：
   violation_rate = slo_violation_count / total_requests
   
2. 调度规则：优先选择违约率最高的客户端的请求执行

3. 优势：
   - 动态调整：根据实时 SLO 违约情况动态调整优先级
   - 改善用户体验：帮助"受苦"用户优先获得服务
   - 类似 VTC：但基于违约率而非 token 消耗

使用场景：
- 需要保证所有用户的 SLO 满足率
- 希望避免某些用户长期得不到好的服务
- 追求整体公平性而非单一指标优化
"""

from experiment.base_experiment import BaseExperiment


class SLOGreedyExperiment(BaseExperiment):
    """
    SLO 违约率贪心调度实验
    
    实现基于 SLO 违约率的优先级调度
    违约率越高的客户端优先级越高
    """
    
    def __init__(self, client):
        super().__init__(client)
        self.exp_type = 'SLOGreedy'
        
        self.logger.info(f"[SLOGreedy] Initialized - prioritize clients with higher SLO violation rate")

