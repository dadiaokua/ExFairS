#!/usr/bin/env python3
"""
测试vLLM引擎调度器队列监控 - 带round time限制

测试内容：
1. 启动真实vLLM引擎
2. 直接向vLLM引擎提交多个请求
3. 监控vLLM调度器中的请求数量变化（等待队列、运行队列、交换队列）
4. 验证round time限制下的异步并发请求处理
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
from config.Config import GLOBAL_CONFIG
from util.PromptLoader import PromptLoader
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


async def collect_generation_output_with_timeout(engine, prompt, sampling_params, request_id, timeout=30):
    """收集生成输出 - 带超时控制"""
    try:
        start_time = time.time()
        logger.info(f"开始处理请求: {request_id}")
        
        # 使用asyncio.wait_for添加超时控制
        async def generate_with_timeout():
            results = []
            async for request_output in engine.generate(prompt, sampling_params, request_id):
                results.append(request_output)
                # 检查是否超时
                if time.time() - start_time > timeout:
                    logger.warning(f"请求 {request_id} 处理超时 ({timeout}s)，停止生成")
                    break
            return results
        
        try:
            results = await asyncio.wait_for(generate_with_timeout(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"请求 {request_id} 超时 ({timeout}s)")
            return {
                'request_id': request_id,
                'status': 'timeout',
                'total_time': timeout,
                'output_tokens': 0
            }
        
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
                'output_tokens': output_tokens,
                'output_text': output_text[:100] + "..." if len(output_text) > 100 else output_text
            }
        else:
            logger.warning(f"请求 {request_id} 没有产生输出")
            return {
                'request_id': request_id,
                'status': 'no_output',
                'total_time': total_time,
                'output_tokens': 0
            }
            
    except Exception as e:
        logger.error(f"请求 {request_id} 处理失败: {e}")
        return {
            'request_id': request_id,
            'status': 'error',
            'total_time': time.time() - start_time,
            'output_tokens': 0,
            'error': str(e)
        }


async def monitor_vllm_scheduler_with_round_time(engine, round_time=30, interval=0.5):
    """监控vLLM调度器状态 - 带round time限制"""
    logger.info(f"开始监控vLLM调度器状态，round time: {round_time}s...")
    
    start_time = time.time()
    max_waiting = 0
    max_running = 0
    max_swapped = 0
    total_samples = 0
    monitoring_data = []
    
    while time.time() - start_time < round_time:
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
                remaining = round_time - elapsed
                
                monitoring_data.append({
                    'elapsed': elapsed,
                    'waiting': waiting_count,
                    'running': running_count,
                    'swapped': swapped_count
                })
                
                logger.info(f"[{elapsed:.1f}s/{round_time}s] vLLM调度器 - 等待: {waiting_count}, 运行: {running_count}, 交换: {swapped_count}, 剩余: {remaining:.1f}s")
                
                # Round time结束前5秒开始警告
                if remaining <= 5 and remaining > 0:
                    total_active = waiting_count + running_count + swapped_count
                    if total_active > 0:
                        logger.warning(f"Round time即将结束，仍有 {total_active} 个请求在处理中")
                        
            else:
                logger.warning("无法访问vLLM调度器")
                
        except Exception as e:
            logger.debug(f"监控调度器状态时出错: {e}")
            
        await asyncio.sleep(interval)
    
    # Round time结束后检查剩余请求
    try:
        if hasattr(engine, 'engine') and hasattr(engine.engine, 'scheduler'):
            scheduler = engine.engine.scheduler
            final_waiting = len(scheduler.waiting) if hasattr(scheduler, 'waiting') else 0
            final_running = len(scheduler.running) if hasattr(scheduler, 'running') else 0
            final_swapped = len(scheduler.swapped) if hasattr(scheduler, 'swapped') else 0
            total_remaining = final_waiting + final_running + final_swapped
            
            logger.info(f"Round time结束 - 未完成请求: 等待={final_waiting}, 运行={final_running}, 交换={final_swapped}, 总计={total_remaining}")
    except Exception as e:
        logger.debug(f"检查最终状态失败: {e}")
    
    logger.info(f"监控完成 - 最大等待: {max_waiting}, 最大运行: {max_running}, 最大交换: {max_swapped}, 总采样: {total_samples}")
    return {
        'max_waiting': max_waiting,
        'max_running': max_running, 
        'max_swapped': max_swapped,
        'total_samples': total_samples,
        'monitoring_data': monitoring_data
    }


async def test_vllm_scheduler_with_round_time():
    """测试vLLM调度器在round time限制下的表现"""
    logger.info("=== 测试vLLM调度器 + Round Time限制 ===")
    
    # 启动vLLM引擎
    engine_manager = VLLMEngineManager()
    try:
        logger.info("启动vLLM引擎...")
        engine = await engine_manager.create_engine(
            model_path="/home/llm/model_hub/Llama-3.1-8B",
            max_num_seqs=8,  # 允许8个并发序列
            tensor_parallel_size=8,
            suppress_logs=True
        )
        
        logger.info("✓ vLLM引擎启动成功")
        
        # 准备测试prompts - 使用较长的prompt确保处理时间
        prompts = [
            "请详细解释人工智能的发展历程，包括从早期符号主义到现代深度学习的演变过程，以及各个阶段的关键技术突破和代表性成果",
            "分析深度学习在计算机视觉领域的应用，详细描述卷积神经网络的工作原理，并举例说明在图像识别、目标检测等任务中的具体实现方法",
            "讨论自然语言处理技术的最新进展，重点介绍Transformer架构和注意力机制的原理，以及在机器翻译、文本生成等任务中的应用效果",
            "解释强化学习的基本概念和算法原理，包括Q-learning、策略梯度等方法，并分析其在游戏AI、机器人控制等领域的成功案例",
            "描述大数据处理和分析技术的发展趋势，包括分布式计算、流处理、实时分析等关键技术，以及在商业智能和决策支持中的应用",
            "分析云计算和边缘计算的技术特点，比较其在不同应用场景下的优劣势，并探讨未来计算架构的发展方向和技术挑战",
            "介绍区块链技术的核心原理和共识机制，分析其在金融科技、供应链管理、数字身份等领域的创新应用和发展前景",
            "讨论量子计算的基本原理和技术优势，解释量子比特、量子纠缠等概念，并分析量子计算在密码学、优化问题等方面的潜在影响",
            "探讨物联网技术的架构设计和关键组件，包括传感器网络、通信协议、数据处理等，以及在智慧城市建设中的具体应用案例",
            "分析网络安全威胁的演变趋势和防护策略，包括恶意软件检测、入侵防护、数据加密等技术，以及安全治理的最佳实践方法"
        ]
        
        # 设置参数
        round_time = 30  # 30秒round time
        request_timeout = round_time - 5  # 请求超时时间比round time短5秒
        sampling_params = create_sampling_params(max_tokens=200)
        
        logger.info(f"Round Time: {round_time}s, 请求超时: {request_timeout}s")
        logger.info(f"准备提交 {len(prompts)} 个请求...")
        
        # 启动监控任务
        monitor_task = asyncio.create_task(
            monitor_vllm_scheduler_with_round_time(engine, round_time=round_time, interval=0.5)
        )
        
        # 快速连续提交所有请求
        tasks = []
        submit_start = time.time()
        
        for i, prompt in enumerate(prompts):
            request_id = f"round_test_{i}_{uuid.uuid4().hex[:6]}"
            
            task = asyncio.create_task(
                collect_generation_output_with_timeout(engine, prompt, sampling_params, request_id, timeout=request_timeout)
            )
            tasks.append(task)
            
            logger.info(f"提交请求 {i+1}: {request_id}")
            await asyncio.sleep(0.1)  # 快速提交
        
        submit_time = time.time() - submit_start
        logger.info(f"✓ 所有请求提交完成，耗时: {submit_time:.3f}s")
        
        # 等待round time结束或所有请求完成（以先到的为准）
        logger.info(f"等待round time ({round_time}s) 结束...")
        
        try:
            # 使用asyncio.wait在round time内等待任务完成
            done, pending = await asyncio.wait(
                tasks, 
                timeout=round_time,
                return_when=asyncio.ALL_COMPLETED
            )
            
            if pending:
                logger.warning(f"Round time结束，仍有 {len(pending)} 个请求未完成，取消这些请求")
                for task in pending:
                    task.cancel()
                
                # 等待取消的任务完成
                await asyncio.gather(*pending, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"等待请求完成时出错: {e}")
        
        # 收集监控结果
        monitor_task.cancel()
        try:
            monitor_stats = await monitor_task
        except asyncio.CancelledError:
            monitor_stats = {'max_waiting': 0, 'max_running': 0, 'max_swapped': 0, 'total_samples': 0}
        
        # 分析结果
        completed_results = []
        for task in tasks:
            if task.done() and not task.cancelled():
                try:
                    result = task.result()
                    if result:
                        completed_results.append(result)
                except Exception as e:
                    logger.debug(f"获取任务结果失败: {e}")
        
        successful_results = [r for r in completed_results if r.get('status') == 'completed']
        timeout_results = [r for r in completed_results if r.get('status') == 'timeout']
        error_results = [r for r in completed_results if r.get('status') == 'error']
        
        logger.info("=== Round Time测试结果分析 ===")
        logger.info(f"Round Time: {round_time}s")
        logger.info(f"总请求数: {len(prompts)}")
        logger.info(f"成功完成: {len(successful_results)}")
        logger.info(f"超时请求: {len(timeout_results)}")
        logger.info(f"错误请求: {len(error_results)}")
        logger.info(f"未完成请求: {len(tasks) - len(completed_results)}")
        logger.info(f"请求提交时间: {submit_time:.3f}s")
        logger.info(f"vLLM最大等待队列: {monitor_stats['max_waiting']}")
        logger.info(f"vLLM最大运行队列: {monitor_stats['max_running']}")
        logger.info(f"vLLM最大交换队列: {monitor_stats['max_swapped']}")
        
        if successful_results:
            avg_time = sum(r['total_time'] for r in successful_results) / len(successful_results)
            avg_tokens = sum(r['output_tokens'] for r in successful_results) / len(successful_results)
            logger.info(f"成功请求平均处理时间: {avg_time:.3f}s")
            logger.info(f"成功请求平均输出tokens: {avg_tokens:.1f}")
        
        # 验证round time控制效果
        assert monitor_stats['max_waiting'] > 0, "vLLM等待队列应该有请求排队"
        assert monitor_stats['max_running'] > 0, "vLLM运行队列应该有请求在处理"
        assert len(completed_results) > 0, "应该至少有一些请求完成"
        
        # 计算吞吐量
        throughput = len(successful_results) / round_time
        logger.info(f"吞吐量: {throughput:.2f} 请求/秒")
        
        logger.info("✓ Round Time控制测试通过")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 清理资源
        await engine_manager.cleanup()


async def test_queue_behavior_under_pressure():
    """测试高负载下的队列行为"""
    logger.info("=== 测试高负载下的队列行为 ===")
    
    engine_manager = VLLMEngineManager()
    try:
        # 使用较小的max_num_seqs来制造队列压力
        engine = await engine_manager.create_engine(
            model_path="/home/llm/model_hub/Llama-3.1-8B",
            max_num_seqs=4,  # 只允许4个并发
            tensor_parallel_size=8,
            suppress_logs=True
        )
        
        logger.info("✓ vLLM引擎启动成功 (max_num_seqs=4)")
        
        # 创建短prompt确保快速处理
        short_prompts = [f"计算 {i} + {i+1} = ?" for i in range(15)]  # 15个简短请求
        
        round_time = 20  # 20秒round time
        sampling_params = create_sampling_params(max_tokens=50)  # 短输出
        
        # 启动监控
        monitor_task = asyncio.create_task(
            monitor_vllm_scheduler_with_round_time(engine, round_time=round_time, interval=0.3)
        )
        
        # 快速提交所有请求
        tasks = []
        for i, prompt in enumerate(short_prompts):
            request_id = f"pressure_test_{i}_{uuid.uuid4().hex[:4]}"
            
            task = asyncio.create_task(
                collect_generation_output_with_timeout(engine, prompt, sampling_params, request_id, timeout=15)
            )
            tasks.append(task)
            
            logger.info(f"提交压力测试请求 {i+1}: {request_id}")
            await asyncio.sleep(0.05)  # 极快提交间隔
        
        logger.info("观察队列压力和处理效率...")
        
        # 等待round time
        done, pending = await asyncio.wait(tasks, timeout=round_time)
        
        if pending:
            logger.warning(f"压力测试中有 {len(pending)} 个请求未完成")
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        
        # 停止监控
        monitor_task.cancel()
        try:
            monitor_stats = await monitor_task
        except asyncio.CancelledError:
            monitor_stats = {'max_waiting': 0, 'max_running': 0, 'max_swapped': 0}
        
        completed_count = len([t for t in tasks if t.done() and not t.cancelled()])
        
        logger.info("=== 压力测试结果 ===")
        logger.info(f"请求总数: {len(short_prompts)}")
        logger.info(f"完成请求: {completed_count}")
        logger.info(f"最大等待队列: {monitor_stats['max_waiting']}")
        logger.info(f"最大运行队列: {monitor_stats['max_running']}")
        logger.info(f"完成率: {completed_count/len(short_prompts)*100:.1f}%")
        
        # 验证压力测试效果
        assert monitor_stats['max_running'] <= 4, f"运行队列不应超过max_num_seqs: {monitor_stats['max_running']}"
        assert monitor_stats['max_waiting'] >= 5, "等待队列应该有明显堆积"
        assert completed_count >= len(short_prompts) * 0.5, "至少应该完成50%的请求"
        
        logger.info("✓ 压力测试通过")
        
    finally:
        await engine_manager.cleanup()


async def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行vLLM调度器队列监控测试套件（带Round Time控制）")
    
    try:
        await test_vllm_scheduler_with_round_time()
        await test_queue_behavior_under_pressure()
        
        logger.info("🎉 所有测试通过！vLLM调度器在Round Time限制下工作正常")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests()) 