"""
Justitia Scheduling Experiment

基于虚拟时间（Virtual Time）的公平调度策略。
使用最小堆（GlobalQueue）维护所有任务的虚拟完成时间，实现真正的公平性。

核心思想：
1. 虚拟时间 V(t) = M / N_t
   - M: 服务器总显存容量
   - N_t: 当前活跃任务数量
   
2. 虚拟完成时间 f_j = V(a_j) + C_j
   - V(a_j): 任务到达时的虚拟时间
   - C_j: 任务的资源需求（通过 MLP 预测）
   
3. 调度规则：始终选择 f_j 最小的任务执行（最小堆维护）

优势：
- 短任务优先：C_j 小的任务 f_j 通常更小，会被优先调度
- 防止饥饿：即使 C_j 大，随着时间推移系统虚拟时间增大，最终也会被调度
"""

from experiment.base_experiment import BaseExperiment


class JustitiaExperiment(BaseExperiment):
    """
    Justitia 调度实验
    
    实现基于虚拟时间的公平调度，使用优先级队列（最小堆）
    按照虚拟完成时间 f_j 进行调度
    """
    
    def __init__(self, client):
        super().__init__(client)
        self.exp_type = 'Justitia'
        
        # Justitia 特有参数
        # M: 服务器总显存容量（单位：KV缓存数量）
        # 这里用一个抽象值表示，实际可以根据GPU显存大小调整
        self.total_memory = 100000  # 假设总容量为 100000 单位
        
        self.logger.info(f"[Justitia] Initialized with total_memory={self.total_memory}")

