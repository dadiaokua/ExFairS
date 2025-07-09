#!/usr/bin/env python3
"""
调试vLLM调度器结构，找到正确的队列访问方法
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


async def debug_scheduler_structure(engine):
    """调试scheduler的结构，找到正确的属性"""
    logger.info("=== 调试vLLM引擎scheduler结构 ===")
    
    try:
        logger.info("引擎类型: " + str(type(engine)))
        
        if hasattr(engine, 'engine'):
            inner_engine = engine.engine
            logger.info("内部引擎类型: " + str(type(inner_engine)))
            
            if hasattr(inner_engine, 'scheduler'):
                scheduler = inner_engine.scheduler
                logger.info("调度器类型: " + str(type(scheduler)))
                
                if isinstance(scheduler, list):
                    logger.info(f"调度器是一个list，长度: {len(scheduler)}")
                    for i, item in enumerate(scheduler):
                        logger.info(f"  调度器[{i}] 类型: {type(item)}")
                        if hasattr(item, '__dict__'):
                            logger.info(f"  调度器[{i}] 属性: {[attr for attr in dir(item) if not attr.startswith('_')]}")
                            
                            # 检查这个调度器对象的队列属性
                            queue_attrs = ['waiting', 'running', 'swapped', 'waiting_queue', 'running_queue']
                            for attr in queue_attrs:
                                if hasattr(item, attr):
                                    value = getattr(item, attr)
                                    logger.info(f"    找到队列属性 {attr}: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                else:
                    logger.info("调度器不是list类型")
                    logger.info("调度器属性: " + str([attr for attr in dir(scheduler) if not attr.startswith('_')]))
            else:
                logger.warning("内部引擎没有scheduler属性")
                
            # 检查是否有其他获取队列信息的方法
            possible_methods = [
                'get_num_unfinished_requests',
                'has_unfinished_requests', 
                'get_scheduler_config'
            ]
            
            for method_name in possible_methods:
                if hasattr(inner_engine, method_name):
                    try:
                        method = getattr(inner_engine, method_name)
                        if callable(method):
                            result = method()
                            logger.info(f"内部引擎方法 {method_name}() 返回: {result}")
                    except Exception as e:
                        logger.info(f"调用 {method_name}() 失败: {e}")
        else:
            logger.warning("引擎没有engine属性")
            
    except Exception as e:
        logger.error(f"调试scheduler结构时出错: {e}")
        import traceback
        traceback.print_exc()


async def test_with_real_requests(engine):
    """在有真实请求运行时检查scheduler状态"""
    logger.info("=== 在真实请求运行时调试scheduler ===")
    
    sampling_params = create_sampling_params(max_tokens=200)
    
    # 提交一些长时间的请求
    tasks = []
    for i in range(3):
        request_id = f"debug_req_{i}_{uuid.uuid4().hex[:6]}"
        prompt = f"请写一篇详细的关于人工智能技术发展的长篇文章，要求至少1000字，包含历史发展、技术原理、应用场景、未来趋势等多个方面的内容。这是第{i+1}个请求。"
        
        task = asyncio.create_task(
            collect_generation_output(engine, prompt, sampling_params, request_id)
        )
        tasks.append(task)
        
        logger.info(f"提交请求 {i+1}: {request_id}")
        await asyncio.sleep(0.2)  # 快速提交
        
        # 立即检查scheduler状态
        await debug_scheduler_state(engine, f"提交请求{i+1}后")
    
    # 等待一段时间再检查
    await asyncio.sleep(1.0)
    await debug_scheduler_state(engine, "等待1秒后")
    
    # 取消任务避免长时间等待
    for task in tasks:
        task.cancel()
    
    await asyncio.gather(*tasks, return_exceptions=True)


async def debug_scheduler_state(engine, context=""):
    """调试当前scheduler状态"""
    try:
        if hasattr(engine, 'engine') and hasattr(engine.engine, 'scheduler'):
            scheduler = engine.engine.scheduler
            
            logger.info(f"[{context}] 调试scheduler状态:")
            
            if isinstance(scheduler, list):
                logger.info(f"  调度器list长度: {len(scheduler)}")
                
                for i, item in enumerate(scheduler):
                    logger.info(f"  === 调度器[{i}] ===")
                    
                    # 尝试不同的访问方法
                    possible_attrs = [
                        'waiting', 'running', 'swapped', 
                        'waiting_queue', 'running_queue', 'swapped_queue',
                        'pending', 'active', 'ready', 'scheduled'
                    ]
                    
                    for attr in possible_attrs:
                        if hasattr(item, attr):
                            try:
                                value = getattr(item, attr)
                                if hasattr(value, '__len__'):
                                    logger.info(f"    {attr}: {len(value)} 个项目")
                                    if len(value) > 0:
                                        logger.info(f"      内容类型: {[type(x) for x in list(value)[:2]]}")
                                else:
                                    logger.info(f"    {attr}: {value}")
                            except Exception as e:
                                logger.info(f"    {attr}: 访问失败 - {e}")
                    
                    # 尝试一些可能的方法
                    possible_methods = [
                        'get_num_unfinished_seq_groups',
                        'get_waiting_queue',
                        'get_running_queue', 
                        'num_batched_tokens'
                    ]
                    
                    for method_name in possible_methods:
                        if hasattr(item, method_name):
                            try:
                                method = getattr(item, method_name)
                                if callable(method):
                                    result = method()
                                    logger.info(f"    {method_name}(): {result}")
                            except Exception as e:
                                logger.info(f"    {method_name}(): 调用失败 - {e}")
            else:
                logger.info("  调度器不是list类型，使用原来的方法")
        
        # 检查引擎级别的方法
        if hasattr(engine, 'engine'):
            inner_engine = engine.engine
            engine_methods = ['get_num_unfinished_requests', 'has_unfinished_requests']
            
            for method_name in engine_methods:
                if hasattr(inner_engine, method_name):
                    try:
                        method = getattr(inner_engine, method_name)
                        if callable(method):
                            result = method()
                            logger.info(f"  引擎方法 {method_name}(): {result}")
                    except Exception as e:
                        logger.info(f"  引擎方法 {method_name}(): 调用失败 - {e}")
                        
    except Exception as e:
        logger.error(f"调试scheduler状态失败: {e}")


async def collect_generation_output(engine, prompt, sampling_params, request_id):
    """收集生成输出"""
    try:
        start_time = time.time()
        results = []
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            results.append(request_output)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if results:
            final_output = results[-1]
            output_tokens = len(final_output.outputs[0].token_ids) if final_output.outputs else 0
            logger.info(f"完成请求: {request_id}, 耗时: {total_time:.3f}s, 输出tokens: {output_tokens}")
            return {'request_id': request_id, 'total_time': total_time, 'output_tokens': output_tokens}
        else:
            logger.warning(f"请求 {request_id} 没有产生输出")
            return None
            
    except Exception as e:
        logger.error(f"请求 {request_id} 处理失败: {e}")
        return None


async def main():
    """主函数"""
    logger.info("开始调试vLLM scheduler结构")
    
    engine_manager = VLLMEngineManager()
    try:
        # 启动引擎，使用较小的并发数便于观察
        engine = await engine_manager.create_engine(
            model_path="/home/llm/model_hub/Llama-3.1-8B",
            max_num_seqs=2,  # 只允许2个并发，容易观察队列
            tensor_parallel_size=8,
            suppress_logs=True
        )
        
        logger.info("✓ vLLM引擎启动成功")
        
        # 调试scheduler结构
        await debug_scheduler_structure(engine)
        
        # 在空闲状态检查
        await debug_scheduler_state(engine, "空闲状态")
        
        # 在有请求时检查
        await test_with_real_requests(engine)
        
    except Exception as e:
        logger.error(f"调试过程失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 