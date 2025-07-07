#!/usr/bin/env python3

import asyncio
import uuid
import logging
from typing import Any
from vllm_engine_helper import VLLMEngineManager, create_sampling_params
from prompt_loader import PromptLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_len(obj: Any) -> int:
    """安全地计算对象长度，处理异常情况"""
    try:
        if isinstance(obj, Exception):
            return 0
        elif hasattr(obj, '__len__'):
            return len(obj)
        elif obj is None:
            return 0
        else:
            return 1
    except:
        return 0

async def collect_generation_output(engine, prompt: str, sampling_params, request_id: str):
    """收集生成输出"""
    results = []
    try:
        logger.debug(f"开始处理请求: {request_id}")
        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)
            # 每收集到一些输出就记录一下
            if len(results) % 10 == 0:
                logger.info(f"请求 {request_id} 已生成 {len(results)} 个输出")
                
        logger.info(f"✓ 请求 {request_id} 完成，总输出: {len(results)}")
        return results
        
    except asyncio.CancelledError:
        logger.info(f"✓ 请求 {request_id} 被取消，已生成: {len(results)} 个输出")
        return results  # 返回已生成的结果，不再抛出异常
    except Exception as e:
        logger.error(f"❌ 请求 {request_id} 出现错误: {e}")
        return results

async def test_multiple_abort_with_engine(engine):
    """测试多个请求的abort功能（使用现有引擎）"""
    logger.info("=== 测试多个请求abort ===")
    
    prompt_loader = PromptLoader()
    
    try:
        # 使用更多token确保请求需要较长时间完成
        sampling_params = create_sampling_params(max_tokens=300)
        
        # 获取多个随机prompts
        prompts = prompt_loader.get_random_prompts(4, "mixed")
        if not prompts:
            prompts = [
                "Write a very detailed and comprehensive explanation of artificial intelligence, including its history, current applications, future prospects, and technical details with examples and code snippets",
                "Create a comprehensive guide to quantum computing that covers all theoretical foundations, practical implementations, and real-world applications with mathematical formulations", 
                "Describe the complete history of space exploration from ancient astronomy to modern missions, including detailed technical specifications and future plans",
                "Write a detailed cookbook for French cuisine with complete recipes, cooking techniques, ingredients, and cultural background for each dish"
            ]
        
        # 启动多个请求
        tasks = []
        request_ids = []
        
        for i, prompt in enumerate(prompts):
            request_id = f"multi_test_{i}_{uuid.uuid4()}"
            request_ids.append(request_id)
            
            task = asyncio.create_task(
                collect_generation_output(engine, prompt, sampling_params, request_id)
            )
            tasks.append(task)
            
            logger.info(f"启动请求 {i+1}: {request_id}")
            await asyncio.sleep(0.2)  # 减少间隔，快速启动
        
        # 等待较短时间，确保请求正在处理但远未完成
        logger.info("等待1秒让请求开始处理...")
        await asyncio.sleep(1)
        
        # 立即abort前两个请求（它们应该还在处理中）
        logger.info("现在abort前两个请求（应该还在处理中）...")
        for i in range(2):
            logger.info(f"尝试abort正在处理的请求: {request_ids[i]}")
            try:
                await engine.abort(request_ids[i])
                logger.info(f"✓ abort调用成功: {request_ids[i]}")
            except Exception as e:
                logger.info(f"❌ abort调用失败: {e}")
        
        # 等待所有任务完成
        logger.info("等待所有任务完成（被abort的应该提前结束）...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("=== 多个请求abort测试完成 ===")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.info(f"请求 {i+1} 异常: {result}")
            else:
                output_count = safe_len(result)
                status = "被abort" if i < 2 and output_count < 250 else "正常完成"
                logger.info(f"请求 {i+1} {status}: {output_count} 个输出")
                
    except Exception as e:
        logger.error(f"多请求abort测试失败: {e}")

async def test_abort_all_methods(engine):
    """测试不同的abort方法"""
    logger.info("=== 测试abort方法 ===")
    
    prompt_loader = PromptLoader()
    
    try:
        # 使用长文本确保有足够时间进行abort
        sampling_params = create_sampling_params(max_tokens=400)
        
        # 启动多个请求
        prompts = [
            "Write a very long and detailed story about space exploration with multiple chapters, character development, and scientific accuracy",
            "Explain quantum mechanics in extreme detail with mathematical proofs, examples, and practical applications",
            "Describe machine learning algorithms comprehensively with code examples, mathematical foundations, and real-world use cases"
        ]
        
        tasks = []
        request_ids = []
        
        for i, prompt in enumerate(prompts):
            request_id = f"abort_test_{i}_{uuid.uuid4()}"
            request_ids.append(request_id)
            
            task = asyncio.create_task(
                collect_generation_output(engine, prompt, sampling_params, request_id)
            )
            tasks.append(task)
            
            logger.info(f"启动请求: {request_id}")
            await asyncio.sleep(0.1)
        
        # 等待很短时间，确保请求开始但远未完成
        logger.info("等待0.5秒让请求开始处理...")
        await asyncio.sleep(0.5)
        
        # 测试AsyncLLMEngine的abort功能
        logger.info("开始测试abort方法（请求应该正在处理中）...")
        
        # 方法1: 使用engine.abort逐个取消
        logger.info("使用engine.abort逐个abort请求...")
        aborted_count = 0
        
        for request_id in request_ids:
            try:
                logger.info(f"尝试abort正在处理的请求: {request_id}")
                await engine.abort(request_id)
                aborted_count += 1
                logger.info(f"✓ 成功abort: {request_id}")
            except Exception as e:
                logger.info(f"❌ abort失败 {request_id}: {e}")
        
        logger.info(f"逐个abort结果: 成功 {aborted_count}/{len(request_ids)} 个请求")
        
        # 方法2: 探索engine.engine属性（既然AsyncLLMEngine有engine属性）
        logger.info("探索engine.engine的功能...")
        try:
            if hasattr(engine, 'engine'):
                inner_engine = engine.engine
                inner_attrs = [attr for attr in dir(inner_engine) if not attr.startswith('_')]
                logger.info(f"inner engine属性数量: {len(inner_attrs)}")
                
                # 查找abort相关方法
                abort_methods = [attr for attr in inner_attrs if 'abort' in attr.lower()]
                if abort_methods:
                    logger.info(f"inner engine的abort方法: {abort_methods}")
                
                # 查找调度器相关
                scheduler_methods = [attr for attr in inner_attrs if 'sched' in attr.lower()]
                if scheduler_methods:
                    logger.info(f"inner engine的调度器方法: {scheduler_methods}")
                    
        except Exception as e:
            logger.error(f"探索inner engine时出错: {e}")
        
        # 等待所有任务完成或被取消
        logger.info("等待所有任务完成（被abort的应该提前结束）...")
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=3)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.info(f"任务 {i+1} 异常: {type(result).__name__}")
                else:
                    output_count = safe_len(result)
                    status = "被abort" if output_count < 300 else "可能完成"
                    logger.info(f"任务 {i+1} {status}: {output_count} 个输出")
        except asyncio.TimeoutError:
            logger.info("等待任务超时，说明abort可能没有生效，取消剩余任务")
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        logger.info("=== abort方法测试完成 ===")
        
    except Exception as e:
        logger.error(f"abort测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_request_id_tracking(engine):
    """测试request ID跟踪和批量abort"""
    logger.info("=== 测试Request ID跟踪和批量abort ===")
    
    prompt_loader = PromptLoader()
    
    # 模拟request ID跟踪
    active_request_ids = set()
    
    try:
        sampling_params = create_sampling_params(max_tokens=350)
        
        # 启动多个请求并跟踪ID
        prompts = [
            "Write a comprehensive analysis of neural networks with detailed mathematical foundations, implementation examples, and practical applications in various industries",
            "Create an in-depth explanation of blockchain technology covering cryptographic principles, consensus mechanisms, and real-world use cases", 
            "Compose a detailed essay about renewable energy technologies including solar, wind, hydro, and emerging technologies with technical specifications",
            "Develop a thorough discussion of artificial intelligence ethics covering bias, privacy, transparency, and societal implications with case studies"
        ]
        
        tasks = []
        
        for i, prompt in enumerate(prompts):
            request_id = f"tracked_test_{i}_{uuid.uuid4()}"
            active_request_ids.add(request_id)  # 跟踪请求ID
            
            task = asyncio.create_task(
                collect_generation_output(engine, prompt, sampling_params, request_id)
            )
            tasks.append(task)
            
            logger.info(f"启动并跟踪请求: {request_id}")
            await asyncio.sleep(0.1)
        
        logger.info(f"当前跟踪的请求ID: {len(active_request_ids)} 个")
        
        # 等待很短时间让请求开始处理但远未完成
        logger.info("等待0.8秒让请求开始处理...")
        await asyncio.sleep(0.8)
        
        # 使用跟踪的ID进行批量abort
        logger.info("开始批量abort跟踪的请求（应该正在处理中）...")
        aborted_count = 0
        failed_requests = set()
        
        # 创建副本避免迭代时修改
        request_ids_to_abort = list(active_request_ids)
        
        for request_id in request_ids_to_abort:
            try:
                logger.info(f"尝试abort正在处理的请求: {request_id}")
                
                # 尝试abort
                success = False
                
                # 方法1: 使用engine.abort (异步)
                if hasattr(engine, 'abort'):
                    try:
                        await engine.abort(request_id)
                        success = True
                        logger.info(f"✓ 使用engine.abort成功: {request_id}")
                    except Exception as e:
                        logger.info(f"❌ engine.abort失败: {e}")
                
                if success:
                    aborted_count += 1
                    active_request_ids.discard(request_id)  # 从跟踪中移除
                else:
                    failed_requests.add(request_id)
                    
            except Exception as e:
                logger.error(f"abort请求 {request_id} 时出现异常: {e}")
                failed_requests.add(request_id)
        
        logger.info(f"批量abort结果: 成功 {aborted_count} 个, 失败 {len(failed_requests)} 个")
        logger.info(f"剩余跟踪的请求: {len(active_request_ids)} 个")
        
        if failed_requests:
            logger.info(f"失败的请求ID: {failed_requests}")
        
        # 等待所有任务完成
        logger.info("等待所有任务完成（被abort的应该提前结束）...")
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.info(f"任务 {i+1} 异常: {type(result).__name__}")
                else:
                    output_count = safe_len(result)
                    status = "被abort" if output_count < 250 else "可能完成"
                    logger.info(f"任务 {i+1} {status}: {output_count} 个输出")
        except asyncio.TimeoutError:
            logger.info("等待任务超时")
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        logger.info("=== Request ID跟踪和批量abort测试完成 ===")
        
    except Exception as e:
        logger.error(f"request ID跟踪测试失败: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """主函数 - 使用单个引擎实例运行所有测试"""
    engine_manager = VLLMEngineManager()
    
    try:
        # 启动一次引擎，供所有测试使用
        logger.info("=== 启动vLLM引擎 ===")
        engine = await engine_manager.start_engine(max_num_seqs=4)
        
        # 运行所有测试
        await test_multiple_abort_with_engine(engine)
        await test_abort_all_methods(engine)  # 现在主要测试逐个abort和引擎探索
        await test_request_id_tracking(engine)
        
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最后关闭引擎
        logger.info("=== 关闭vLLM引擎 ===")
        await engine_manager.shutdown_engine()

if __name__ == "__main__":
    asyncio.run(main()) 