#!/usr/bin/env python3
"""
优先级策略的理论依据演示
展示为什么需要基于排名的相对优先级而不是绝对优先级
"""

import asyncio
import time
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Request:
    id: str
    priority: int
    submit_time: float
    client_type: str

class PriorityStrategyDemo:
    def __init__(self):
        self.requests = []
        self.priority_distribution = {}
    
    def add_request(self, request: Request):
        """添加请求到队列"""
        self.requests.append(request)
        self.priority_distribution[request.priority] = self.priority_distribution.get(request.priority, 0) + 1
    
    def strict_priority_insert(self, new_request: Request) -> int:
        """严格优先级策略：直接插到所有低优先级请求前面"""
        for i, existing in enumerate(self.requests):
            if existing.priority > new_request.priority:
                return i
        return len(self.requests)
    
    def rank_based_priority_insert(self, new_request: Request) -> int:
        """基于排名的相对优先级策略"""
        if not self.priority_distribution:
            return len(self.requests)
        
        # 计算优先级排名
        all_priorities = sorted(self.priority_distribution.keys())
        higher_priority_count = len([p for p in all_priorities if p < new_request.priority])
        
        if new_request.priority in all_priorities:
            priority_rank_ratio = higher_priority_count / len(all_priorities)
        else:
            priority_rank_ratio = higher_priority_count / (len(all_priorities) + 1)
        
        # 计算可超越的请求数量
        can_overtake_count = 0
        for existing_priority, count in self.priority_distribution.items():
            if existing_priority > new_request.priority:
                can_overtake_count += count
        
        # 计算优先级优势和前进位置
        priority_advantage = 1 - priority_rank_ratio
        base_forward_positions = int(can_overtake_count * priority_advantage)
        max_forward_positions = min(base_forward_positions, can_overtake_count, len(self.requests))
        
        # 计算插入位置
        insert_pos = max(0, len(self.requests) - max_forward_positions)
        
        return insert_pos
    
    def simulate_scenario(self, scenario_name: str, requests: List[Request]):
        """模拟特定场景"""
        print(f"\n=== {scenario_name} ===")
        
        # 重置状态
        self.requests = []
        self.priority_distribution = {}
        
        # 添加初始请求
        for req in requests[:-1]:  # 除了最后一个
            self.add_request(req)
        
        new_request = requests[-1]  # 最后一个是要插入的新请求
        
        print(f"初始队列: {[(r.id, r.priority) for r in self.requests]}")
        print(f"优先级分布: {self.priority_distribution}")
        print(f"要插入的新请求: {new_request.id}(priority={new_request.priority})")
        
        # 严格优先级策略
        strict_pos = self.strict_priority_insert(new_request)
        strict_queue = self.requests.copy()
        strict_queue.insert(strict_pos, new_request)
        
        # 基于排名的策略
        rank_pos = self.rank_based_priority_insert(new_request)
        rank_queue = self.requests.copy()
        rank_queue.insert(rank_pos, new_request)
        
        print(f"\n严格优先级结果 (位置 {strict_pos}):")
        print(f"  队列: {[(r.id, r.priority) for r in strict_queue]}")
        
        print(f"\n基于排名的结果 (位置 {rank_pos}):")
        print(f"  队列: {[(r.id, r.priority) for r in rank_queue]}")
        
        # 分析差异
        jumped_strict = len(self.requests) - strict_pos
        jumped_rank = len(self.requests) - rank_pos
        
        print(f"\n分析:")
        print(f"  严格优先级：跳过 {jumped_strict} 个请求")
        print(f"  基于排名：跳过 {jumped_rank} 个请求")
        print(f"  差异：基于排名策略更温和，减少了 {jumped_strict - jumped_rank} 个跳跃")

def demonstrate_starvation_problem():
    """演示饥饿问题"""
    print("=" * 60)
    print("演示：为什么需要相对优先级而不是绝对优先级")
    print("=" * 60)
    
    demo = PriorityStrategyDemo()
    
    # 场景1：饥饿问题
    scenario1 = [
        Request("A", 10, time.time(), "normal"),
        Request("B", 10, time.time(), "normal"),
        Request("C", 10, time.time(), "normal"),
        Request("D", 10, time.time(), "normal"),
        Request("E", 10, time.time(), "normal"),
        Request("HIGH", 5, time.time(), "urgent"),  # 高优先级请求
    ]
    
    demo.simulate_scenario("场景1：单个高优先级请求", scenario1)
    
    # 场景2：多样化优先级环境
    scenario2 = [
        Request("A", 5, time.time(), "urgent"),
        Request("B", 10, time.time(), "normal"),
        Request("C", 5, time.time(), "urgent"),
        Request("D", 15, time.time(), "low"),
        Request("E", 10, time.time(), "normal"),
        Request("F", 20, time.time(), "batch"),
        Request("NEW", 8, time.time(), "medium"),  # 中等优先级请求
    ]
    
    demo.simulate_scenario("场景2：多样化优先级环境", scenario2)
    
    # 场景3：极端情况
    scenario3 = [
        Request("L1", 20, time.time(), "low"),
        Request("L2", 20, time.time(), "low"),
        Request("L3", 20, time.time(), "low"),
        Request("L4", 20, time.time(), "low"),
        Request("L5", 20, time.time(), "low"),
        Request("URGENT", 1, time.time(), "critical"),  # 最高优先级
    ]
    
    demo.simulate_scenario("场景3：极端优先级差异", scenario3)

def explain_mathematical_foundation():
    """解释数学基础"""
    print("\n" + "=" * 60)
    print("数学基础和理论依据")
    print("=" * 60)
    
    print("""
1. 排名归一化 (Rank Normalization):
   priority_rank_ratio = higher_priority_count / total_priority_levels
   
   这个公式将优先级排名归一化到 [0, 1] 区间：
   - 0 表示最高优先级
   - 1 表示最低优先级
   
2. 优先级优势 (Priority Advantage):
   priority_advantage = 1 - priority_rank_ratio
   
   这是一个反比例关系：
   - 排名越高(ratio越小) -> 优势越大(advantage越大)
   - 排名越低(ratio越大) -> 优势越小(advantage越小)
   
3. 比例插队 (Proportional Queue Jumping):
   forward_positions = can_overtake_count * priority_advantage
   
   这确保了：
   - 高优先级请求能够获得更多插队机会
   - 但插队数量与其优先级地位成比例
   - 避免了绝对的"全部跳过"
   
4. 公平性保证 (Fairness Guarantee):
   - 每个优先级都有相应的处理机会
   - 优先级差异体现在处理速度上，而非绝对的处理顺序
   - 符合"比例公平"的资源分配原则
   
5. 系统稳定性 (System Stability):
   - 避免了优先级反转导致的系统不稳定
   - 防止了低优先级请求的无限期等待
   - 提供了可预测的性能表现
    """)

def analyze_real_world_applications():
    """分析现实世界的应用"""
    print("\n" + "=" * 60)
    print("现实世界的应用场景")
    print("=" * 60)
    
    print("""
这个策略的理论依据来自多个领域：

1. 操作系统调度 (OS Scheduling):
   - Linux CFS (Completely Fair Scheduler) 使用类似的比例公平思想
   - 避免进程饥饿问题
   
2. 网络流量控制 (Network Traffic Control):
   - QoS (Quality of Service) 中的加权公平队列
   - 保证不同优先级流量的相对公平性
   
3. 资源分配理论 (Resource Allocation Theory):
   - 经济学中的帕累托最优
   - 博弈论中的纳什均衡
   
4. 排队论 (Queueing Theory):
   - M/M/1 队列的优先级扩展
   - 防止高优先级请求导致的系统不稳定
   
5. 机器学习推理服务:
   - 不同客户端的SLA要求
   - 批处理vs实时请求的平衡
   - 资源利用率的优化
    """)

if __name__ == "__main__":
    demonstrate_starvation_problem()
    explain_mathematical_foundation()
    analyze_real_world_applications() 