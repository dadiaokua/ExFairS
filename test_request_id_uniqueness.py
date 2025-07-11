#!/usr/bin/env python3
"""
测试request_id生成的唯一性
"""
import asyncio
import concurrent.futures
import time
import threading
from collections import defaultdict

# 导入我们的唯一ID生成函数
try:
    from RequestQueueManager.RequestQueueManager import generate_unique_request_id
except ImportError:
    print("无法导入generate_unique_request_id，使用备用实现")
    import uuid
    
    _request_counter = 0
    _counter_lock = threading.Lock()
    
    def generate_unique_request_id(client_id: str, worker_id: str) -> str:
        """生成唯一的请求ID，避免重复"""
        global _request_counter
        with _counter_lock:
            _request_counter += 1
            counter = _request_counter
        
        # 使用UUID确保全局唯一性，并添加可读的前缀
        unique_id = str(uuid.uuid4())[:8]  # 取UUID的前8位
        timestamp = int(time.time() * 1000000)  # 使用微秒时间戳
        
        return f"req_{client_id}_{worker_id}_{counter}_{timestamp}_{unique_id}"


def test_sequential_generation():
    """测试顺序生成request_id的唯一性"""
    print("=== 测试顺序生成 ===")
    
    request_ids = []
    for i in range(1000):
        request_id = generate_unique_request_id(f"client_{i%10}", f"worker_{i%5}")
        request_ids.append(request_id)
    
    # 检查重复
    unique_ids = set(request_ids)
    duplicates = len(request_ids) - len(unique_ids)
    
    print(f"生成了 {len(request_ids)} 个request_id")
    print(f"唯一的request_id: {len(unique_ids)}")
    print(f"重复的request_id: {duplicates}")
    
    if duplicates > 0:
        # 找出重复的ID
        id_counts = defaultdict(int)
        for rid in request_ids:
            id_counts[rid] += 1
        
        print("重复的request_id:")
        for rid, count in id_counts.items():
            if count > 1:
                print(f"  {rid}: {count} 次")
    
    return duplicates == 0


def test_concurrent_generation():
    """测试并发生成request_id的唯一性"""
    print("\n=== 测试并发生成 ===")
    
    def generate_batch(thread_id, batch_size=100):
        """每个线程生成一批request_id"""
        batch_ids = []
        for i in range(batch_size):
            request_id = generate_unique_request_id(f"client_{thread_id}", f"worker_{i}")
            batch_ids.append(request_id)
        return batch_ids
    
    # 使用多线程并发生成
    num_threads = 10
    batch_size = 100
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        futures = [executor.submit(generate_batch, i, batch_size) for i in range(num_threads)]
        
        # 收集所有结果
        all_request_ids = []
        for future in concurrent.futures.as_completed(futures):
            batch_ids = future.result()
            all_request_ids.extend(batch_ids)
    
    # 检查重复
    unique_ids = set(all_request_ids)
    duplicates = len(all_request_ids) - len(unique_ids)
    
    print(f"生成了 {len(all_request_ids)} 个request_id")
    print(f"唯一的request_id: {len(unique_ids)}")
    print(f"重复的request_id: {duplicates}")
    
    if duplicates > 0:
        # 找出重复的ID
        id_counts = defaultdict(int)
        for rid in all_request_ids:
            id_counts[rid] += 1
        
        print("重复的request_id:")
        for rid, count in id_counts.items():
            if count > 1:
                print(f"  {rid}: {count} 次")
    
    return duplicates == 0


async def test_async_generation():
    """测试异步生成request_id的唯一性"""
    print("\n=== 测试异步生成 ===")
    
    async def generate_async_batch(client_id, batch_size=100):
        """异步生成一批request_id"""
        batch_ids = []
        for i in range(batch_size):
            request_id = generate_unique_request_id(f"client_{client_id}", f"worker_{i}")
            batch_ids.append(request_id)
            # 模拟异步操作
            await asyncio.sleep(0.001)  # 1ms延迟
        return batch_ids
    
    # 并发运行多个异步任务
    num_tasks = 10
    batch_size = 50
    
    tasks = [generate_async_batch(i, batch_size) for i in range(num_tasks)]
    results = await asyncio.gather(*tasks)
    
    # 合并所有结果
    all_request_ids = []
    for batch_ids in results:
        all_request_ids.extend(batch_ids)
    
    # 检查重复
    unique_ids = set(all_request_ids)
    duplicates = len(all_request_ids) - len(unique_ids)
    
    print(f"生成了 {len(all_request_ids)} 个request_id")
    print(f"唯一的request_id: {len(unique_ids)}")
    print(f"重复的request_id: {duplicates}")
    
    if duplicates > 0:
        # 找出重复的ID
        id_counts = defaultdict(int)
        for rid in all_request_ids:
            id_counts[rid] += 1
        
        print("重复的request_id:")
        for rid, count in id_counts.items():
            if count > 1:
                print(f"  {rid}: {count} 次")
    
    return duplicates == 0


def test_high_frequency_generation():
    """测试高频率生成request_id的唯一性"""
    print("\n=== 测试高频率生成 ===")
    
    def rapid_generation():
        """快速生成request_id"""
        request_ids = []
        start_time = time.time()
        
        # 在很短时间内生成大量ID
        for i in range(1000):
            request_id = generate_unique_request_id("client_rapid", f"worker_{i%10}")
            request_ids.append(request_id)
        
        end_time = time.time()
        print(f"生成1000个ID耗时: {end_time - start_time:.6f}秒")
        
        return request_ids
    
    # 多线程同时快速生成
    num_threads = 5
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(rapid_generation) for _ in range(num_threads)]
        
        all_request_ids = []
        for future in concurrent.futures.as_completed(futures):
            batch_ids = future.result()
            all_request_ids.extend(batch_ids)
    
    # 检查重复
    unique_ids = set(all_request_ids)
    duplicates = len(all_request_ids) - len(unique_ids)
    
    print(f"总共生成了 {len(all_request_ids)} 个request_id")
    print(f"唯一的request_id: {len(unique_ids)}")
    print(f"重复的request_id: {duplicates}")
    
    return duplicates == 0


def main():
    """运行所有测试"""
    print("开始测试request_id生成的唯一性...")
    
    # 运行各种测试
    test_results = []
    
    test_results.append(test_sequential_generation())
    test_results.append(test_concurrent_generation())
    test_results.append(asyncio.run(test_async_generation()))
    test_results.append(test_high_frequency_generation())
    
    # 汇总结果
    print("\n=== 测试结果汇总 ===")
    test_names = ["顺序生成", "并发生成", "异步生成", "高频率生成"]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(test_results)
    print(f"\n总体结果: {'✓ 所有测试通过' if all_passed else '✗ 存在测试失败'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 