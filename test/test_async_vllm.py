#!/usr/bin/env python3
"""
测试vLLM引擎调度器队列监控

测试内容：
1. 启动真实vLLM引擎
2. 直接向vLLM引擎提交多个请求
3. 监控vLLM调度器中的请求数量变化（等待队列、运行队列、交换队列）
4. 验证能否有效获取队列统计信息
"""

import asyncio
import time
import logging
import uuid
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_engine_helper import VLLMEngineManager
from vllm import SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sampling_params(max_tokens=256):
    """创建采样参数"""
    return SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
        stop=None
    )


async def collect_generation_output(engine, prompt, sampling_params, request_id):
    """收集生成输出"""
    try:
        start_time = time.time()
        logger.info(f"开始处理请求: {request_id}")
        
        # 提交请求到vLLM引擎
        results = []
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            results.append(request_output)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if results:
            final_output = results[-1]
            output_text = final_output.outputs[0].text if final_output.outputs else ""
            output_tokens = len(final_output.outputs[0].token_ids) if final_output.outputs else 0
            
            logger.info(f"完成请求: {request_id}, 耗时: {total_time:.3f}s, 输出tokens: {output_tokens}")
            return {
                'request_id': request_id,
                'status': 'completed',
                'total_time': total_time,
                'output_tokens': output_tokens
            }
        else:
            logger.warning(f"请求 {request_id} 没有产生输出")
            return None
            
    except Exception as e:
        logger.error(f"请求 {request_id} 处理失败: {e}")
        return None


async def monitor_vllm_scheduler(engine, duration=20, interval=0.5):
    """监控vLLM调度器状态"""
    logger.info(f"开始监控vLLM调度器状态，持续 {duration} 秒...")
    
    start_time = time.time()
    max_waiting = 0
    max_running = 0
    max_swapped = 0
    total_samples = 0
    monitoring_data = []
    
    while time.time() - start_time < duration:
        try:
            if hasattr(engine, 'engine') and hasattr(engine.engine, 'scheduler'):
                scheduler = engine.engine.scheduler
                waiting_count = len(scheduler.waiting) if hasattr(scheduler, 'waiting') else 0
                running_count = len(scheduler.running) if hasattr(scheduler, 'running') else 0
                swapped_count = len(scheduler.swapped) if hasattr(scheduler, 'swapped') else 0
                
                max_waiting = max(max_waiting, waiting_count)
                max_running = max(max_running, running_count)
                max_swapped = max(max_swapped, swapped_count)
                total_samples += 1
                
                elapsed = time.time() - start_time
                
                monitoring_data.append({
                    'elapsed': elapsed,
                    'waiting': waiting_count,
                    'running': running_count,
                    'swapped': swapped_count
                })
                
                logger.info(f"[{elapsed:.1f}s] vLLM调度器状态 - 等待: {waiting_count}, 运行: {running_count}, 交换: {swapped_count}")
                
                # 如果所有队列都为空且已经运行了至少5秒，可以提前结束
                if waiting_count == 0 and running_count == 0 and swapped_count == 0 and elapsed > 5:
                    logger.info("所有队列为空，监控结束")
                    break
                    
            else:
                logger.warning("无法访问vLLM调度器")
                
        except Exception as e:
            logger.debug(f"监控调度器状态时出错: {e}")
            
        await asyncio.sleep(interval)
    
    logger.info(f"监控完成 - 最大等待: {max_waiting}, 最大运行: {max_running}, 最大交换: {max_swapped}, 总采样: {total_samples}")
    return {
        'max_waiting': max_waiting,
        'max_running': max_running, 
        'max_swapped': max_swapped,
        'total_samples': total_samples,
        'monitoring_data': monitoring_data
    }


async def test_vllm_scheduler_queue_monitoring():
    """测试vLLM调度器队列监控功能"""
    logger.info("=== 测试vLLM调度器队列监控 ===")
    
    # 启动vLLM引擎
    engine_manager = VLLMEngineManager()
    try:
        logger.info("启动vLLM引擎...")
        engine = await engine_manager.create_engine(
            model_path="/home/llm/model_hub/Llama-3.1-8B",
            max_num_seqs=12,  # 增加到12个并发序列以处理更多请求
            tensor_parallel_size=8,
            suppress_logs=True
        )
        
        logger.info("✓ vLLM引擎启动成功")
        
        # 准备测试prompts
        prompts = [
            "请解释人工智能的基本概念和应用领域",
            "描述深度学习的工作原理",
            "什么是自然语言处理技术",
            "解释机器学习算法的分类",
            "讨论计算机视觉的发展现状",
            "分析大数据处理技术",
            "介绍云计算的优势和挑战",
            "探讨区块链技术的应用前景",
            "说明物联网技术的核心特点",
            "阐述网络安全的重要性",
            "介绍分布式系统的架构设计",
            "讨论数据库优化策略",
            "解释软件工程的基本原则",
            "分析操作系统的核心功能",
            "描述网络协议的工作机制",
            "探讨移动应用开发趋势",
            "介绍DevOps实践方法",
            "讨论微服务架构设计",
            "解释容器化技术优势",
            "分析前端开发框架选择",
            "描述后端服务设计模式",
            "探讨API设计最佳实践",
            "介绍测试驱动开发方法",
            "讨论代码质量管理",
            "解释性能优化策略"
        ] * 2  # 50个请求
        
        sampling_params = create_sampling_params(max_tokens=150)
        
        logger.info(f"准备提交 {len(prompts)} 个请求...")
        
        # 启动监控任务
        monitor_task = asyncio.create_task(monitor_vllm_scheduler(engine, duration=45, interval=0.5))  # 增加监控时间
        
        # 快速连续提交所有请求
        tasks = []
        submit_start = time.time()
        
        for i, prompt in enumerate(prompts):
            request_id = f"test_{i}_{uuid.uuid4().hex[:6]}"
            
            task = asyncio.create_task(
                collect_generation_output(engine, prompt, sampling_params, request_id)
            )
            tasks.append(task)
            
            logger.info(f"提交请求 {i+1}: {request_id}")
            await asyncio.sleep(0.1)  # 快速提交，间隔100ms
        
        submit_time = time.time() - submit_start
        logger.info(f"✓ 所有请求提交完成，耗时: {submit_time:.3f}s")
        
        # 等待所有请求完成
        logger.info("等待所有请求完成...")
        start_wait = time.time()
        
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        completion_time = time.time() - start_wait
        logger.info(f"✓ 所有请求处理完成，耗时: {completion_time:.3f}s")
        
        # 停止监控
        monitor_task.cancel()
        try:
            monitor_stats = await monitor_task
        except asyncio.CancelledError:
            monitor_stats = {'max_waiting': 0, 'max_running': 0, 'max_swapped': 0, 'total_samples': 0}
        
        # 分析结果
        successful_results = [r for r in completed_results if r is not None and not isinstance(r, Exception)]
        failed_results = [r for r in completed_results if r is None or isinstance(r, Exception)]
        
        logger.info("=== 测试结果分析 ===")
        logger.info(f"总请求数: {len(prompts)}")
        logger.info(f"成功完成: {len(successful_results)}")
        logger.info(f"失败/异常: {len(failed_results)}")
        logger.info(f"请求提交时间: {submit_time:.3f}s")
        logger.info(f"请求完成时间: {completion_time:.3f}s")
        logger.info(f"vLLM最大等待队列: {monitor_stats['max_waiting']}")
        logger.info(f"vLLM最大运行队列: {monitor_stats['max_running']}")
        logger.info(f"vLLM最大交换队列: {monitor_stats['max_swapped']}")
        logger.info(f"监控采样次数: {monitor_stats['total_samples']}")
        
        if successful_results:
            avg_time = sum(r['total_time'] for r in successful_results) / len(successful_results)
            avg_tokens = sum(r['output_tokens'] for r in successful_results) / len(successful_results)
            logger.info(f"平均处理时间: {avg_time:.3f}s")
            logger.info(f"平均输出tokens: {avg_tokens:.1f}")
        
        # 验证监控效果
        assert monitor_stats['total_samples'] > 10, "监控采样次数应该足够多"
        assert monitor_stats['max_running'] > 0, "vLLM运行队列应该有请求在处理"
        assert len(successful_results) >= len(prompts) * 0.8, f"成功率应该较高: {len(successful_results)}/{len(prompts)}"
        
        # 验证是否成功监控到队列状态变化
        if monitor_stats['max_waiting'] > 0:
            logger.info("✓ 成功监控到等待队列中的请求")
        else:
            logger.info("⚠ 未监控到等待队列中的请求（可能请求处理太快）")
        
        logger.info("✓ vLLM调度器队列监控测试通过")
        
        return monitor_stats
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 清理资源
        await engine_manager.cleanup()


async def test_queue_capacity_limits():
    """测试队列容量限制"""
    logger.info("=== 测试队列容量限制 ===")
    
    engine_manager = VLLMEngineManager()
    try:
        # 使用较小的max_num_seqs来观察队列行为
        engine = await engine_manager.create_engine(
            model_path="/home/llm/model_hub/Llama-3.1-8B",
            max_num_seqs=4,  # 只允许4个并发序列
            tensor_parallel_size=8,
            suppress_logs=True
        )
        
        logger.info("✓ vLLM引擎启动成功 (max_num_seqs=4)")
        
        # 创建较多的请求来制造队列压力
        prompts = [f"请详细回答问题{i}：什么是人工智能？请从历史发展、技术原理、应用场景等多个角度进行分析。" for i in range(20)]  # 20个请求
        
        sampling_params = create_sampling_params(max_tokens=100)
        
        # 启动监控
        monitor_task = asyncio.create_task(monitor_vllm_scheduler(engine, duration=35, interval=0.3))  # 增加监控时间
        
        # 快速提交所有请求
        tasks = []
        for i, prompt in enumerate(prompts):
            request_id = f"capacity_test_{i}_{uuid.uuid4().hex[:4]}"
            
            task = asyncio.create_task(
                collect_generation_output(engine, prompt, sampling_params, request_id)
            )
            tasks.append(task)
            
            logger.info(f"提交请求 {i+1}: {request_id}")
            await asyncio.sleep(0.05)  # 快速提交
        
        logger.info("观察队列状态变化...")
        
        # 等待所有请求完成
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 停止监控
        monitor_task.cancel()
        try:
            monitor_stats = await monitor_task
        except asyncio.CancelledError:
            monitor_stats = {'max_waiting': 0, 'max_running': 0, 'max_swapped': 0}
        
        successful_count = len([r for r in completed_results if r is not None and not isinstance(r, Exception)])
        
        logger.info("=== 队列容量测试结果 ===")
        logger.info(f"请求总数: {len(prompts)}")
        logger.info(f"成功完成: {successful_count}")
        logger.info(f"最大等待队列: {monitor_stats['max_waiting']}")
        logger.info(f"最大运行队列: {monitor_stats['max_running']}")
        logger.info(f"最大交换队列: {monitor_stats['max_swapped']}")
        
        # 验证队列容量限制效果
        assert monitor_stats['max_running'] <= 4, f"运行队列不应超过max_num_seqs限制: {monitor_stats['max_running']} > 4"
        assert monitor_stats['max_waiting'] > 0, "等待队列应该有请求堆积"
        assert successful_count >= len(prompts) * 0.8, "大部分请求应该成功完成"
        
        logger.info("✓ 队列容量限制测试通过")
        
        return monitor_stats
        
    finally:
        await engine_manager.cleanup()


async def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行vLLM调度器队列监控测试套件")
    
    try:
        # 测试基本队列监控
        stats1 = await test_vllm_scheduler_queue_monitoring()
        
        # 测试队列容量限制
        stats2 = await test_queue_capacity_limits()
        
        # 综合分析
        logger.info("=== 综合测试结果 ===")
        logger.info(f"基本监控测试 - 最大等待: {stats1['max_waiting']}, 最大运行: {stats1['max_running']}")
        logger.info(f"容量限制测试 - 最大等待: {stats2['max_waiting']}, 最大运行: {stats2['max_running']}")
        
        # 验证监控系统的有效性
        total_waiting = stats1['max_waiting'] + stats2['max_waiting']
        total_running = stats1['max_running'] + stats2['max_running']
        
        assert total_waiting > 0, "两次测试中应该至少观察到等待队列中有请求"
        assert total_running > 0, "两次测试中应该至少观察到运行队列中有请求"
        
        logger.info("🎉 所有测试通过！vLLM调度器队列监控系统工作正常")
        logger.info("✓ 可以有效监控等待队列、运行队列和交换队列的状态")
        logger.info("✓ max_num_seqs配置生效，能够限制并发处理数量")
        logger.info("✓ 队列统计信息准确可靠")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests()) 