import asyncio
import time
from datetime import datetime
from typing import Any
import uuid

import numpy as np
import logging

import random
import threading
from vllm import SamplingParams
from config.Config import GLOBAL_CONFIG
from util.ThreadSafeUtil import ThreadSafeCounter


async def process_stream(stream):
    first_token_time = None
    total_tokens = 0
    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        if chunk.choices[0].delta.content:
            total_tokens += 1
        if chunk.choices[0].finish_reason is not None:
            break
    return first_token_time, total_tokens


async def make_request(client, experiment, request, start_time=None, request_id=None, priority=None):
    """
    发送请求 - 自动检测使用直接引擎还是HTTP客户端
    """
    # 检查是否有直接的vLLM引擎
    if 'vllm_engine' in GLOBAL_CONFIG and GLOBAL_CONFIG['vllm_engine'] is not None:
        # 使用直接引擎API
        return await make_request_direct_engine(GLOBAL_CONFIG['vllm_engine'], experiment, request, start_time,
                                                request_id, priority)
    else:
        # 使用HTTP客户端（原有方式）
        return await make_request_http_client(client, experiment, request, start_time, request_id)


async def make_request_direct_engine(engine, experiment, request, start_time=None, request_id=None, priority=None):
    """
    直接使用AsyncLLMEngine处理请求
    
    Args:
        engine: AsyncLLMEngine实例
        experiment: 实验对象
        request: 请求内容
        start_time: 开始时间
        request_id: 请求ID（如果提供则使用，否则生成新的）
        
    Returns:
        tuple: (output_tokens, elapsed_time, tokens_per_second, ttft, input_tokens, slo_met)
        :param priority:
    """
    if start_time is None:
        start_time = time.time()

    # 如果没有提供request_id，生成一个带客户端前缀的唯一ID
    if request_id is None:
        client_id = getattr(experiment.client, 'client_id', 'unknown_client')
        request_id = f"{client_id}_{str(uuid.uuid4())}"

    try:
        # 注册请求ID到实验的客户端（如果可用）
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'register_request_id'):
            experiment.client.register_request_id(request_id)

        # 添加调试日志
        client_id = getattr(experiment.client, 'client_id', 'unknown_client')
        experiment.logger.debug(f"Client {client_id}: 开始处理请求 {request_id}")

        # 准备请求参数
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=experiment.output_tokens
        )

        # 发送请求到引擎 - 注意这里返回的是异步生成器
        request_generator = engine.generate(request, sampling_params, request_id, priority=priority)

        # 使用async for迭代异步生成器获取最终结果
        request_output = None
        first_chunk_time = None

        async for output_chunk in request_generator:
            if first_chunk_time is None:
                first_chunk_time = time.time()
            request_output = output_chunk  # 保留最后一个输出作为最终结果

        end_time = time.time()

        # 计算首个token时间 (TTFT)
        ttft_ms = None
        if first_chunk_time:
            ttft_ms = (first_chunk_time - start_time) * 1000  # 转换为毫秒

        # 如果没有获取到任何输出，返回None
        if request_output is None:
            experiment.logger.warning(f"Client {client_id}: 请求 {request_id} 没有返回任何输出")
            if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
                experiment.client.unregister_request_id(request_id)
            return None

        # 获取输出内容
        output_text = ""
        input_tokens = 0
        output_tokens = 0

        if hasattr(request_output, 'outputs') and request_output.outputs:
            output = request_output.outputs[0]
            output_text = output.text
            output_tokens = len(output.token_ids) if hasattr(output, 'token_ids') else 0

        # 获取输入token数（如果可用）
        if hasattr(request_output, 'prompt_token_ids'):
            input_tokens = len(request_output.prompt_token_ids)
        elif hasattr(request_output, 'inputs') and hasattr(request_output.inputs, 'token_ids'):
            input_tokens = len(request_output.inputs.token_ids)

        elapsed_time = end_time - start_time
        tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0

        # 检查SLO
        slo_met = elapsed_time * 1000 <= experiment.latency_slo  # 转换为毫秒

        # 添加完成日志
        experiment.logger.debug(
            f"Client {client_id}: 完成请求 {request_id}, 耗时: {elapsed_time:.3f}s, 输出tokens: {output_tokens}")

        # 取消注册请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        return output_tokens, elapsed_time, tokens_per_second, ttft_ms, input_tokens, 1 if slo_met else 0

    except asyncio.TimeoutError:
        end_time = time.time()
        # 记录timeout次数
        if hasattr(experiment, 'timeout_count'):
            experiment.timeout_count += 1

        # 超时时也要注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        experiment.logger.warning(
            f"Client {experiment.client_id} request timed out after {end_time - start_time} seconds (Total timeouts: {experiment.timeout_count})")

        return None
    except Exception as e:
        # 异常时也要注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        experiment.logger.error(f"Error during direct engine request: {str(e)}")
        return None


async def make_request_http_client(client, experiment, request, start_time=None, request_id=None):
    """
    使用HTTP客户端处理请求（原有方式）
    """
    if start_time is None:
        start_time = time.time()

    # 如果没有提供request_id，生成一个
    if request_id is None:
        request_id = str(uuid.uuid4())

    try:
        # 注册请求ID到实验的客户端（如果可用）
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'register_request_id'):
            experiment.client.register_request_id(request_id)

        # 使用log_request=False参数来禁止在日志中打印请求内容
        stream = await client.chat.completions.create(
            model=GLOBAL_CONFIG['request_model_name'],
            messages=[{"role": "user", "content": request}],
            max_tokens=experiment.output_tokens,
            stream=True
            # 注意：移除 extra_headers，因为 OpenAI 客户端可能不支持
            # 请求ID仍然会被跟踪，但不会通过header传递给服务器
        )
        first_token_time, output_tokens = await asyncio.wait_for(process_stream(stream),
                                                                 timeout=experiment.request_timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        input_token = experiment.tokenizer(request, truncation=False, return_tensors="pt").input_ids[0]
        tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0

        # 正常完成时注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        return output_tokens, elapsed_time, tokens_per_second, ttft, len(
            input_token), 1 if elapsed_time <= experiment.latency_slo else 0

    except asyncio.TimeoutError:
        end_time = time.time()
        # 记录timeout次数
        if hasattr(experiment, 'timeout_count'):
            experiment.timeout_count += 1

        # 超时时也要注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        experiment.logger.warning(
            f"Client {experiment.client_id} request timed out after {end_time - start_time} seconds (Total timeouts: {experiment.timeout_count})")

        return None
    except Exception as e:
        # 异常时也要注销请求ID
        if hasattr(experiment, 'client') and hasattr(experiment.client, 'unregister_request_id'):
            experiment.client.unregister_request_id(request_id)

        experiment.logger.error(f"Error during request: {str(e)}")
        return None


async def make_request_via_queue(queue_manager, client_id: str, worker_id: str,
                                 request_content: str, experiment, priority: int = 0, request_id: str = None) -> Any:
    """通过队列管理器发送请求"""
    # 如果没有提供request_id，生成一个（向后兼容）
    if request_id is None:
        request_id = str(uuid.uuid4())

    try:
        # 提交请求到队列
        submitted_request_id = await queue_manager.submit_request(
            client_id=client_id,
            worker_id=worker_id,
            request_content=request_content,
            experiment=experiment,
            priority=priority,
            start_time=time.time(),
            request_id=request_id  # 传递预生成的request_id
        )

        # 等待响应
        result = await queue_manager.get_response(client_id, timeout=1000)

        return result

    except Exception as e:
        experiment.logger.error(f"Error making request via queue: {e}")
        return None


def calculate_all_request_times(experiment, qmp_per_worker):
    """
    预先计算所有请求的时间点
    
    Args:
        experiment: 实验对象，包含round_time, distribution, time_ratio等属性
        qmp_per_worker: 每个worker每分钟发送的请求数量
    
    Returns:
        list: 请求时间点列表
    """
    # 从experiment对象中获取参数
    rate_lambda = qmp_per_worker
    round_time = experiment.round_time
    distribution = experiment.distribution
    time_ratio = experiment.time_ratio

    # 预留缓冲时间给最后的请求完成
    buffer_time = round_time * GLOBAL_CONFIG.get('buffer_ratio', 0.5)
    # 确保缓冲时间不超过round_time的50%
    buffer_time = min(buffer_time, round_time * 0.5)
    # 实际可用的发送时间窗口
    effective_round_time = round_time - buffer_time

    # 将每分钟请求数转换为每秒请求数
    rate_per_second = rate_lambda / 60.0

    if rate_per_second <= 0:
        rate_per_second = 0.001

    # 基础时间间隔
    base_interval = 1 / rate_per_second

    # 估算总请求数，基于完整的round_time，而不是effective_round_time
    # 这样即使有buffer time，我们仍然会发送完整的请求数量
    estimated_requests = int(round_time * rate_per_second)

    # 移除随机变化
    # random_variation = random.uniform(0.9, 1.1)
    # estimated_requests = int(estimated_requests * random_variation)

    # 生成所有请求的时间点
    request_times = []
    global_start_time = time.time()  # 使用当前时间作为全局开始时间

    # 移除随机开始偏移
    # start_offset = random.uniform(0, min(5.0, effective_round_time * 0.1))
    start_offset = 0  # 从0开始，没有随机偏移

    # 先生成基础时间点（相对于开始时间的偏移）
    base_times = []
    current_offset = start_offset  # 从偏移开始

    # 计算时间压缩比例，将round_time的请求压缩到effective_round_time内
    compression_ratio = effective_round_time / round_time if round_time > 0 else 1.0

    for i in range(estimated_requests):
        # 根据分布类型计算间隔，但移除额外的随机性
        if distribution.lower() == "poisson":
            # 泊松分布
            # base_rate_variation = random.uniform(0.7, 1.3)
            # adjusted_interval = base_interval * base_rate_variation
            interval = float(np.random.exponential(base_interval))
        elif distribution.lower() == "normal":
            # 正态分布，使用固定标准差
            std_dev = base_interval * 0.3  # 固定标准差为间隔的30%
            interval = base_interval + float(np.random.normal(0, std_dev))
            interval = max(0.001, interval)  # 确保间隔为正
        else:
            # 均匀分布，不添加额外变化
            # variation_range = random.uniform(0.3, 0.7)
            # interval = base_interval + float(np.random.uniform(-base_interval * variation_range,
            #                                                   base_interval * variation_range))
            interval = base_interval  # 使用固定间隔
            # interval = max(0.001, interval)

        # 应用压缩比例，将请求间隔压缩
        compressed_interval = interval * compression_ratio

        current_offset += compressed_interval
        if current_offset > effective_round_time:  # 确保不超出有效时间窗口
            break
        base_times.append(current_offset)

        # shuffled_all_request_times(base_times, start_offset, effective_round_time, time_ratio, global_start_time, experiment, buffer_time)

    for base_time in base_times:
        request_time = global_start_time + base_time
        request_times.append(request_time)

    # 记录生成的请求数量
    experiment.logger.info(
        f"Generated {len(request_times)} requests for QPM {qmp_per_worker} in {effective_round_time:.1f}s effective window (buffer: {buffer_time:.1f}s)")

    # 保存请求时间列表到文件
    save_request_times_to_file(experiment, request_times, qmp_per_worker, global_start_time)

    return request_times


def save_request_times_to_file(experiment, request_times, qmp_per_worker, global_start_time):
    """将请求时间列表保存到文件中，按客户端分组，累积记录模式"""
    import os
    import json
    from datetime import datetime
    
    # 确保目录存在
    os.makedirs('tmp_result', exist_ok=True)
    
    # 获取文件名
    timestamp = GLOBAL_CONFIG.get("monitor_file_time", datetime.now().strftime("%m_%d_%H_%M"))
    filename = f'tmp_result/request_times_{timestamp}.json'
    
    # 获取客户端信息
    client_id = getattr(experiment.client, 'client_id', 'unknown_client')
    client_type = getattr(experiment.client, 'client_type', 'unknown')
    worker_id = getattr(experiment, 'worker_id', 'unknown_worker')
    
    # 获取轮次信息
    round_num = getattr(experiment, 'config_round', 0)
    
    # 转换时间戳为相对时间（便于分析）
    relative_times = [t - global_start_time for t in request_times]
    
    # 准备要保存的数据
    client_round_data = {
        'client_id': client_id,
        'client_type': client_type,
        'worker_id': worker_id,
        'round_num': round_num,
        'qmp_per_worker': qmp_per_worker,
        'round_time': experiment.round_time,
        'distribution': experiment.distribution,
        'time_ratio': experiment.time_ratio,
        'global_start_time': global_start_time,
        'global_start_time_24h': datetime.fromtimestamp(global_start_time).strftime("%Y-%m-%d %H:%M:%S"),
        'total_requests': len(request_times),
        'request_times': {
            'absolute_timestamps': request_times,  # 绝对时间戳
            'relative_seconds': relative_times,    # 相对于开始时间的秒数
            'first_request_at': relative_times[0] if relative_times else 0,
            'last_request_at': relative_times[-1] if relative_times else 0,
            'time_span': relative_times[-1] - relative_times[0] if len(relative_times) > 1 else 0
        },
        'statistics': {
            'avg_interval': sum(relative_times[i+1] - relative_times[i] for i in range(len(relative_times)-1)) / max(len(relative_times)-1, 1) if len(relative_times) > 1 else 0,
            'min_interval': min(relative_times[i+1] - relative_times[i] for i in range(len(relative_times)-1)) if len(relative_times) > 1 else 0,
            'max_interval': max(relative_times[i+1] - relative_times[i] for i in range(len(relative_times)-1)) if len(relative_times) > 1 else 0
        },
        'recorded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 读取现有文件或创建新文件
    all_data = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_data = {}
    
    # 如果文件是新的，初始化结构
    if 'metadata' not in all_data:
        all_data['metadata'] = {
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'experiment_type': getattr(experiment, 'exp_type', 'unknown'),
            'total_clients': 0,
            'total_rounds': 0
        }
        all_data['experiments'] = {}
    
    # 创建唯一的实验键：client_id + round_num
    experiment_key = f"{client_id}_round_{round_num}"
    
    # 添加实验数据
    all_data['experiments'][experiment_key] = client_round_data
    
    # 更新元数据
    unique_clients = set()
    unique_rounds = set()
    for exp_key, exp_data in all_data['experiments'].items():
        unique_clients.add(exp_data['client_id'])
        unique_rounds.add(exp_data['round_num'])
    
    all_data['metadata']['total_clients'] = len(unique_clients)
    all_data['metadata']['total_rounds'] = len(unique_rounds)
    all_data['metadata']['total_experiments'] = len(all_data['experiments'])
    all_data['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存到文件
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        experiment.logger.info(f"Request times saved to {filename} for {experiment_key}")
        experiment.logger.info(f"  - Total requests: {len(request_times)}")
        experiment.logger.info(f"  - Time span: {relative_times[-1] - relative_times[0]:.2f}s" if len(relative_times) > 1 else "  - Single request")
        experiment.logger.info(f"  - Average interval: {client_round_data['statistics']['avg_interval']:.3f}s")
        experiment.logger.info(f"  - File now contains {len(all_data['experiments'])} experiments from {len(unique_clients)} clients")
        
    except Exception as e:
        experiment.logger.error(f"Failed to save request times to {filename}: {e}")


def shuffled_all_request_times(base_times, start_offset, effective_round_time, time_ratio, global_start_time,
                               experiment, buffer_time):
    # 对时间点进行轻微的随机打散，但保持总体顺序
    shuffled_times = []
    for i, base_offset in enumerate(base_times):
        # 添加小幅随机偏移
        jitter = random.uniform(-0.5, 0.5)  # ±0.5秒的抖动
        jittered_offset = base_offset + jitter
        jittered_offset = max(start_offset, min(effective_round_time, jittered_offset))  # 确保在有效范围内

        # 应用非线性映射（在有效时间窗口内）
        if time_ratio > 1 and jittered_offset <= effective_round_time:
            # 使用sigmoid类函数进行平滑映射
            progress = jittered_offset / effective_round_time
            # 调整后的进度，保持开始和结束点不变，但中间部分根据time_ratio拉伸
            adjusted_progress = progress ** (1 / time_ratio)
            adjusted_offset = adjusted_progress * effective_round_time
        else:
            # time_ratio <= 1的情况，直接线性缩放
            adjusted_offset = jittered_offset * time_ratio

        # 确保调整后的时间不会超出有效时间窗口
        if adjusted_offset > effective_round_time:
            adjusted_offset = effective_round_time

        # 将偏移转换为绝对时间
        request_time = global_start_time + adjusted_offset
        shuffled_times.append(request_time)

    # 最后再次打散一部分时间点，增加随机性
    if len(shuffled_times) > 1:
        # 随机选择20%的时间点进行轻微重排
        num_to_shuffle = max(1, len(shuffled_times) // 5)
        indices_to_shuffle = random.sample(range(len(shuffled_times)), num_to_shuffle)

        # 对选中的时间点进行局部随机化（确保仍在有效时间窗口内）
        for idx in indices_to_shuffle:
            # 在附近范围内随机调整时间
            if idx > 0 and idx < len(shuffled_times) - 1:
                min_time = (shuffled_times[idx - 1] + shuffled_times[idx]) / 2
                max_time = (shuffled_times[idx] + shuffled_times[idx + 1]) / 2
                # 确保不超出有效时间窗口
                max_time = min(max_time, global_start_time + effective_round_time)
                if min_time < max_time:
                    shuffled_times[idx] = random.uniform(min_time, max_time)

    # 确保时间点仍然是递增的
    shuffled_times.sort()

    # 最终检查：确保所有时间点都在有效窗口内
    shuffled_times = [t for t in shuffled_times if t <= global_start_time + effective_round_time]

    experiment.logger.info(
        f"Generated {len(shuffled_times)} request times in {effective_round_time}s window, buffer: {buffer_time}s")
    if shuffled_times:
        last_request_time = shuffled_times[-1] - global_start_time
        experiment.logger.info(
            f"Last request at: {last_request_time:.2f}s, buffer remaining: {effective_round_time - last_request_time:.2f}s")

    return shuffled_times


async def worker(experiment, selected_clients, semaphore, results, worker_id, worker_json, qmp_per_worker):
    """每个task发送单个请求，使用预先计算的时间点控制间隔"""
    assert worker_json is not None, "sample_content is None!"
    assert isinstance(worker_json, list), f"sample_content is not a list! type={type(worker_json)}"
    assert len(worker_json) > 0, "sample_content is empty!"

    global_start_time = time.time()
    request_count = 0
    drift_time = 0
    completed = 0
    tasks = []
    task_status = {}

    # 创建线程安全的token计数器
    tokens_counter = ThreadSafeCounter()

    # 获取客户端ID用于日志
    client_id = getattr(experiment.client, 'client_id', f'unknown_client_worker_{worker_id}')

    # 预先计算所有请求的时间点
    request_times = calculate_all_request_times(experiment, qmp_per_worker)

    for target_time in request_times:
        if time.time() - global_start_time >= experiment.round_time:
            break
        current_time = time.time()
        if target_time <= current_time:
            # 如果目标时间已过，直接发送请求
            drift_time = current_time - target_time
        else:
            # 如果还没到目标时间，先sleep
            sleep_time = target_time - current_time
            if sleep_time > 0:
                sleep_start = time.time()
                await asyncio.sleep(sleep_time)
            else:
                experiment.logger.warning(
                    f"Client {client_id}: Warning: Negative sleep time detected: {sleep_time:.6f} seconds")
                continue

        # 发送请求（不管是否需要sleep，都会执行到这里）
        request = random.choice(worker_json)
        selected_client = selected_clients[worker_id % len(selected_clients)]

        # 生成request_id并在创建task时就集成
        request_id = f"{client_id}_{str(uuid.uuid4())}"

        task = asyncio.create_task(
            process_request(selected_client, experiment, request, worker_id, results, semaphore, tokens_counter,
                            request_id)
        )

        # 在task_status中存储request_id和其他信息
        task_status[task] = {
            "request_id": request_id,
            "start_time": time.time(),
            "status": "running"
        }

        # 更新done回调以正确更新状态
        def make_done_callback(task_ref):
            def callback(t):
                if task_ref in task_status:
                    try:
                        # 检查task是否成功完成并返回了有效结果
                        if t.cancelled():
                            status = "cancelled"
                        elif t.exception():
                            status = "failed"
                        else:
                            result = t.result()
                            # 如果process_request返回了有效结果（不是None），标记为真正的completed
                            # 否则标记为failed，这样abort时可以识别这些失败的请求
                            status = "completed" if result is not None else "failed"

                        task_status[task_ref].update({
                            "status": status,
                            "end_time": time.time()
                        })
                    except Exception as e:
                        # 异常情况下标记为failed
                        task_status[task_ref].update({
                            "status": "failed",
                            "end_time": time.time()
                        })

            return callback

        task.add_done_callback(make_done_callback(task))
        tasks.append(task)
        request_count += 1

    elapsed = time.time() - global_start_time
    remaining_time = experiment.round_time - elapsed
    if remaining_time > experiment.round_time * GLOBAL_CONFIG.get('buffer_ratio', 0.5) * 1.1:
        if remaining_time > (experiment.round_time * GLOBAL_CONFIG.get('buffer_ratio', 0.5) * 2):
            experiment.logger.warning(
                f"Client {client_id}: Warning: Not enough requests to fill the round time. Sleeping for {remaining_time:.2f} seconds")
        await asyncio.sleep(remaining_time)
    else:
        experiment.logger.info(f"Client {client_id}: reached the end of the round time.")

    # 将task_status存储到experiment中，供abort使用
    if hasattr(experiment, 'client'):
        experiment.client.task_status = task_status

    # 等待所有任务完成
    if tasks:
        # 等待取消操作完成
        # await asyncio.gather(*tasks, return_exceptions=True)

        # 计算总耗时
        total_elapsed_time = time.time() - global_start_time

        completed = sum(1 for status in task_status.values() if status["status"] == "completed")
        cancelled_count = sum(1 for task in tasks if task.cancelled())

        experiment.logger.info(
            f"Client {client_id} Worker {worker_id}: Total tasks: {request_count}, Completed: {completed}, Cancelled: {cancelled_count}")
        experiment.logger.info(
            f"Client {client_id} Worker {worker_id}: Task completion rate: {completed / len(tasks) * 100:.2f}%")
        experiment.logger.info(f"Client {client_id} Worker {worker_id}: Total tokens processed: {tokens_counter.value}")
        experiment.logger.info(
            f"Client {client_id} Worker {worker_id}: Total elapsed time: {total_elapsed_time:.2f} seconds, Round time: {experiment.round_time:.2f} seconds, More than round time: {total_elapsed_time - experiment.round_time:.2f} seconds")

        for task in tasks:
            task.cancel()

    return completed, drift_time, request_count


async def process_request(client, experiment, request, worker_id, results, semaphore, tokens_counter, request_id=None):
    # 如果没有提供request_id，生成一个（向后兼容）
    if request_id is None:
        request_id = str(uuid.uuid4())

    async with semaphore:
        try:
            # 检查当前token总数是否超限
            if hasattr(experiment, 'max_tokens') and tokens_counter.value >= experiment.max_tokens:
                experiment.logger.info(f"Worker {worker_id} reached max tokens limit ({experiment.max_tokens})")
                return

            result = await make_request(client, experiment, request, request_id=request_id)
            if result:
                output_tokens = result[0]  # 第一个元素是output_tokens
                # 原子性地更新token计数
                new_total = tokens_counter.add(output_tokens)
                results.append(result)

                # 如果超过限制，记录日志
                if hasattr(experiment, 'max_tokens') and new_total >= experiment.max_tokens:
                    experiment.logger.info(
                        f"Worker {worker_id} reached max tokens after processing: {new_total}/{experiment.max_tokens}")

        except Exception as e:
            logging.error(
                f"Worker {worker_id} {experiment.config_round + 1} round for client {experiment.client_index} raised an exception: {e}")


async def worker_with_queue(experiment, queue_manager, semaphore, results, worker_id, worker_json, qmp_per_worker):
    """使用队列管理器的worker函数"""
    assert worker_json is not None, "sample_content is None!"
    assert isinstance(worker_json, list), f"sample_content is not a list! type={type(worker_json)}"
    assert len(worker_json) > 0, "sample_content is empty!"

    global_start_time = time.time()
    request_count = 0
    drift_time = 0
    completed = 0
    tasks = []
    task_status = {}

    # 创建线程安全的token计数器
    tokens_counter = ThreadSafeCounter()

    # 注册客户端到队列管理器
    client_id = f"{experiment.client_id}_worker_{worker_id}"
    # 移除直接注册，submit_request会自动处理
    # await queue_manager.register_client(client_id, experiment.client.client_type)

    # 获取客户端ID用于日志
    main_client_id = getattr(experiment.client, 'client_id', 'unknown_client')

    # 预先计算所有请求的时间点
    request_times = calculate_all_request_times(experiment, qmp_per_worker)

    for target_time in request_times:
        if time.time() - global_start_time >= experiment.round_time:
            break
        current_time = time.time()
        if target_time <= current_time:
            # 如果目标时间已过，直接发送请求
            drift_time = current_time - target_time
        else:
            # 如果还没到目标时间，先sleep
            sleep_time = target_time - current_time
            if sleep_time > 0:
                sleep_start = time.time()
                await asyncio.sleep(sleep_time)
            else:
                experiment.logger.warning(
                    f"Client {main_client_id}: Warning: Negative sleep time detected: {sleep_time:.6f} seconds")
                continue

        # 发送请求到队列（不管是否需要sleep，都会执行到这里）
        request = random.choice(worker_json)

        # 设置优先级（短请求优先级更高）
        priority = experiment.client.priority

        # 生成request_id并在创建task时就集成
        request_id = f"{main_client_id}_{str(uuid.uuid4())}"

        task = asyncio.create_task(
            process_request_with_queue(queue_manager, client_id, experiment, request,
                                       worker_id, results, semaphore, tokens_counter,
                                       priority, request_id)
        )

        # 在task_status中存储request_id和其他信息
        task_status[task] = {
            "request_id": request_id,
            "start_time": time.time(),
            "status": "running"
        }

        # 更新done回调以正确更新状态
        def make_done_callback(task_ref):
            def callback(t):
                if task_ref in task_status:
                    try:
                        # 检查task是否成功完成并返回了有效结果
                        if t.cancelled():
                            status = "cancelled"
                        elif t.exception():
                            status = "failed"
                        else:
                            result = t.result()
                            # 如果process_request返回了有效结果（不是None），标记为真正的completed
                            # 否则标记为failed，这样abort时可以识别这些失败的请求
                            status = "completed" if result is not None else "failed"

                        task_status[task_ref].update({
                            "status": status,
                            "end_time": time.time()
                        })
                    except Exception as e:
                        # 异常情况下标记为failed
                        task_status[task_ref].update({
                            "status": "failed",
                            "end_time": time.time()
                        })

            return callback

        task.add_done_callback(make_done_callback(task))
        tasks.append(task)
        request_count += 1

    elapsed = time.time() - global_start_time
    remaining_time = experiment.round_time - elapsed
    if remaining_time > experiment.round_time * GLOBAL_CONFIG.get('buffer_ratio', 0.5) * 1.1:  # 只在剩余时间大于3秒时才sleep，防止误差
        await asyncio.sleep(remaining_time)
    else:
        experiment.logger.info(f"Client {main_client_id}: reached the end of the round time.")

    # 将task_status存储到experiment中，供abort使用
    if hasattr(experiment, 'client'):
        experiment.client.task_status = task_status

    # 等待所有任务完成
    if tasks:
        # await asyncio.gather(*tasks, return_exceptions=True)

        # 计算总耗时
        total_elapsed_time = time.time() - global_start_time

        completed = sum(1 for status in task_status.values() if status["status"] == "completed")
        cancelled_count = sum(1 for task in tasks if task.cancelled())

        experiment.logger.info(
            f"Client {main_client_id} Worker {worker_id}: Total tasks: {request_count}, Completed: {completed}, Cancelled: {cancelled_count}")
        experiment.logger.info(
            f"Client {main_client_id} Worker {worker_id}: Task completion rate: {completed / len(tasks) * 100:.2f}%")
        experiment.logger.info(
            f"Client {main_client_id} Worker {worker_id}: Total tokens processed: {tokens_counter.value}")
        experiment.logger.info(
            f"Client {main_client_id} Worker {worker_id}: Total elapsed time: {total_elapsed_time:.2f} seconds, Round time: {experiment.round_time:.2f} seconds, More than round time: {total_elapsed_time - experiment.round_time:.2f} seconds")

        for task in tasks:
            task.cancel()

    return completed, drift_time, request_count


async def process_request_with_queue(queue_manager, client_id, experiment, request, worker_id, results, semaphore,
                                     tokens_counter, priority=0, request_id=None):
    """使用队列管理器处理请求"""
    # 如果没有提供request_id，生成一个（向后兼容）
    if request_id is None:
        request_id = str(uuid.uuid4())

    async with semaphore:
        try:
            # 检查当前token总数是否超限
            if hasattr(experiment, 'max_tokens') and tokens_counter.value >= experiment.max_tokens:
                experiment.logger.info(f"Worker {worker_id} reached max tokens limit ({experiment.max_tokens})")
                return

            result = await make_request_via_queue(
                queue_manager, client_id, f"worker_{worker_id}",
                request, experiment, priority, request_id
            )

            if result:
                output_tokens = result[0]  # 第一个元素是output_tokens
                # 原子性地更新token计数
                new_total = tokens_counter.add(output_tokens)
                results.append(result)

                # 如果超过限制，记录日志
                if hasattr(experiment, 'max_tokens') and new_total >= experiment.max_tokens:
                    experiment.logger.info(
                        f"Worker {worker_id} reached max tokens after processing: {new_total}/{experiment.max_tokens}")

        except Exception as e:
            logging.error(
                f"Worker {worker_id} {experiment.config_round + 1} round for client {experiment.client_index} raised an exception: {e}")
