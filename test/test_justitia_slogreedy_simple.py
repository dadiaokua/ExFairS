#!/usr/bin/env python3
"""
简化的 Justitia 和 SLO Greedy 测试
不依赖完整项目环境，仅测试核心逻辑
"""

import heapq


class TestJustitiaStrategy:
    """测试 Justitia 策略"""
    
    def __init__(self):
        self.total_memory = 100000
        self.heap = []
        
    def calculate_virtual_time(self, active_tasks):
        """计算虚拟时间 V(t) = M / N_t"""
        return self.total_memory / max(active_tasks, 1)
    
    def estimate_cost(self, input_tokens, output_tokens):
        """估算任务成本 C_j"""
        return input_tokens + 2 * output_tokens
    
    def add_request(self, request_id, input_tokens, output_tokens):
        """添加请求到堆"""
        active_tasks = len(self.heap) + 1
        virtual_time = self.calculate_virtual_time(active_tasks)
        cost = self.estimate_cost(input_tokens, output_tokens)
        virtual_finish_time = virtual_time + cost
        
        heapq.heappush(self.heap, (virtual_finish_time, request_id))
        
        print(f"[Justitia Test] Added request {request_id}:")
        print(f"  - Active tasks: {active_tasks}")
        print(f"  - V(t): {virtual_time:.2f}")
        print(f"  - Cost C_j: {cost}")
        print(f"  - Virtual finish time f_j: {virtual_finish_time:.2f}")
        print()
        
        return virtual_finish_time
    
    def get_next_request(self):
        """从堆中获取下一个请求"""
        if self.heap:
            virtual_finish_time, request_id = heapq.heappop(self.heap)
            print(f"[Justitia Test] Selected request {request_id} with f_j={virtual_finish_time:.2f}")
            print(f"  - Remaining tasks: {len(self.heap)}")
            print()
            return request_id
        return None
    
    def run_test(self):
        """运行测试"""
        print("=" * 60)
        print("Testing Justitia Virtual Time Scheduling")
        print("=" * 60)
        print()
        
        # 模拟不同长度的请求
        requests = [
            ("req_short_1", 100, 50),      # 短请求: cost = 100 + 2*50 = 200
            ("req_long_1", 500, 300),      # 长请求: cost = 500 + 2*300 = 1100
            ("req_short_2", 120, 60),      # 短请求: cost = 120 + 2*60 = 240
            ("req_medium_1", 300, 150),    # 中等请求: cost = 300 + 2*150 = 600
            ("req_short_3", 80, 40),       # 短请求: cost = 80 + 2*40 = 160
        ]
        
        print("Request costs (input + 2*output):")
        for req_id, input_tok, output_tok in requests:
            cost = input_tok + 2 * output_tok
            print(f"  - {req_id}: {cost}")
        print()
        
        # 添加所有请求
        for req_id, input_tok, output_tok in requests:
            self.add_request(req_id, input_tok, output_tok)
        
        print("-" * 60)
        print("Scheduling order (should prioritize short jobs):")
        print("-" * 60)
        print()
        
        # 按照虚拟完成时间顺序取出
        scheduled_order = []
        while self.heap:
            req_id = self.get_next_request()
            scheduled_order.append(req_id)
        
        print("Final scheduling order:")
        for i, req_id in enumerate(scheduled_order, 1):
            print(f"  {i}. {req_id}")
        print()
        
        # 验证短任务优先
        assert "short" in scheduled_order[0], "First task should be short"
        print("✅ Short job prioritization verified")
        print()
        
        print("=" * 60)
        print("✅ Justitia test completed successfully")
        print("=" * 60)
        print()


class TestSLOGreedyStrategy:
    """测试 SLO Greedy 策略"""
    
    def __init__(self):
        self.heap = []
        self.client_stats = {}
    
    def update_client_stats(self, client_id, total_requests, slo_violations):
        """更新客户端统计"""
        self.client_stats[client_id] = {
            'total_requests': total_requests,
            'slo_violations': slo_violations,
            'violation_rate': slo_violations / max(total_requests, 1)
        }
    
    def add_request(self, request_id, client_id):
        """添加请求到堆"""
        stats = self.client_stats.get(client_id, {'total_requests': 1, 'slo_violations': 0})
        violation_rate = stats['slo_violations'] / max(stats['total_requests'], 1)
        
        # 使用负的违约率，因为 heapq 是最小堆
        priority = -violation_rate
        
        heapq.heappush(self.heap, (priority, request_id, client_id))
        
        print(f"[SLO Greedy Test] Added request {request_id}:")
        print(f"  - Client: {client_id}")
        print(f"  - Violation rate: {violation_rate:.3f} ({violation_rate*100:.1f}%)")
        print(f"  - Priority (negative): {priority:.3f}")
        print()
    
    def get_next_request(self):
        """从堆中获取下一个请求"""
        if self.heap:
            neg_rate, request_id, client_id = heapq.heappop(self.heap)
            violation_rate = -neg_rate
            print(f"[SLO Greedy Test] Selected request {request_id}:")
            print(f"  - Client: {client_id}")
            print(f"  - Violation rate: {violation_rate:.3f} ({violation_rate*100:.1f}%)")
            print(f"  - Remaining requests: {len(self.heap)}")
            print()
            return (request_id, client_id)
        return None
    
    def run_test(self):
        """运行测试"""
        print("=" * 60)
        print("Testing SLO Greedy Scheduling")
        print("=" * 60)
        print()
        
        # 设置不同客户端的SLO违约情况
        self.update_client_stats("client_good", total_requests=100, slo_violations=5)    # 5% 违约率
        self.update_client_stats("client_medium", total_requests=100, slo_violations=15) # 15% 违约率
        self.update_client_stats("client_bad", total_requests=100, slo_violations=30)    # 30% 违约率
        
        print("Client statistics:")
        for client_id, stats in sorted(self.client_stats.items()):
            print(f"  - {client_id}: {stats['violation_rate']:.1%} violation rate "
                  f"({stats['slo_violations']}/{stats['total_requests']})")
        print()
        
        # 添加来自不同客户端的请求
        requests = [
            ("req_1", "client_good"),
            ("req_2", "client_bad"),
            ("req_3", "client_medium"),
            ("req_4", "client_good"),
            ("req_5", "client_bad"),
        ]
        
        for req_id, client_id in requests:
            self.add_request(req_id, client_id)
        
        print("-" * 60)
        print("Scheduling order (should prioritize high violation rate clients):")
        print("-" * 60)
        print()
        
        # 按照违约率顺序取出（高违约率优先）
        scheduled_order = []
        while self.heap:
            result = self.get_next_request()
            if result:
                scheduled_order.append(result)
        
        print("Final scheduling order:")
        for i, (req_id, client_id) in enumerate(scheduled_order, 1):
            rate = self.client_stats[client_id]['violation_rate']
            print(f"  {i}. {req_id} from {client_id} ({rate:.1%} violation rate)")
        print()
        
        # 验证高违约率客户端优先
        first_client = scheduled_order[0][1]
        first_rate = self.client_stats[first_client]['violation_rate']
        print(f"First scheduled client: {first_client} with {first_rate:.1%} violation rate")
        assert first_rate >= 0.15, "First client should have high violation rate"
        print("✅ High violation rate prioritization verified")
        print()
        
        print("=" * 60)
        print("✅ SLO Greedy test completed successfully")
        print("=" * 60)
        print()


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Justitia & SLO Greedy Test Suite" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    # 测试 Justitia
    justitia_test = TestJustitiaStrategy()
    justitia_test.run_test()
    
    # 测试 SLO Greedy
    slogreedy_test = TestSLOGreedyStrategy()
    slogreedy_test.run_test()
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "All Tests Passed! 🎉" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")


if __name__ == "__main__":
    main()

