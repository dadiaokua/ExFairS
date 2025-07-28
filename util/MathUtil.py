import os

import numpy as np

from config.Config import GLOBAL_CONFIG
from util.BaseUtil import selectClients_LFS, selectClients_VTC, exchange_resources

import datetime

from util.FileSaveUtil import save_to_file


def calculate_Jains_index(clients, exp_type):
    """
    Calculates Jain's Fairness Index for a list of clients based on their fairness_ratio.
    Since fairness_ratio is "smaller is better", we first normalize the ratios,
    then use (1 - normalized_ratio) to calculate the index.
    Logs the calculation details with a timestamp to a file.
    """
    timestamp = datetime.datetime.now().isoformat()
    log_entry_prefix = f"[{timestamp}] "

    fairness_ratio = [client.fairness_ratio for client in clients]  # Corrected typo: clienr -> client
    n = len(fairness_ratio)

    log_message = f"{log_entry_prefix}Calculating Jain's Index for {n} clients. Original Fairness Ratios: {fairness_ratio}. "

    if n == 0:
        j = 0  # Avoid division by zero, define result as 0 for no clients
        log_message += f"Result: {j} (n=0)."
    elif n == 1:
        # For single client, fairness index is perfect (1.0)
        j = 1.0
        log_message += f"Result: {j} (single client, perfect fairness)."
    else:
        # Step 1: Normalize fairness ratios
        min_ratio = min(fairness_ratio)
        max_ratio = max(fairness_ratio)

        log_message += f"Min ratio: {min_ratio}, Max ratio: {max_ratio}. "

        if max_ratio == min_ratio:
            # All ratios are equal, perfect fairness
            normalized_ratios = [0.0] * n  # All normalized to 0
            transformed_ratios = [1.0] * n  # All transformed to 1
            log_message += f"All ratios equal, perfect fairness. Normalized ratios: {normalized_ratios}, Transformed ratios: {transformed_ratios}. "
        else:
            # Normalize to [0, 1] range: (ratio - min) / (max - min)
            normalized_ratios = [(ratio - min_ratio) / (max_ratio - min_ratio) for ratio in fairness_ratio]
            log_message += f"Normalized ratios: {normalized_ratios}. "

            # Step 2: Transform normalized ratios (1 - normalized_ratio)
            # Since smaller fairness_ratio is better, we want larger transformed values for better performance
            transformed_ratios = [1 - normalized_ratio for normalized_ratio in normalized_ratios]
            log_message += f"Transformed ratios (1 - normalized): {transformed_ratios}. "

        # Step 3: Calculate Jain's Index using transformed ratios
        sum_service = sum(transformed_ratios)
        sum_squares = sum(s ** 2 for s in transformed_ratios)
        denominator = n * sum_squares

        log_message += f"Sum(transformed_ratios): {sum_service}, Sum(transformed_ratios^2): {sum_squares}, Denominator (n * Sum(transformed_ratios^2)): {denominator}. "

        if denominator == 0:
            # Handle division by zero. This should not happen with proper transformed ratios,
            # but we handle it for safety
            j = 0
            log_message += f"Result: {j} (Denominator is zero - unexpected case)."
        else:
            j = (sum_service ** 2) / denominator
            log_message += f"Result: {j}."

    # Append the log message to the file
    try:
        # Define the log directory and file path
        LOG_DIR = "tmp_result"
        LOG_FILE = os.path.join(LOG_DIR,
                                f"{exp_type}_jains_index_calculation_log_{GLOBAL_CONFIG['monitor_file_time']}.log")

        # Ensure the log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)
        save_to_file(LOG_FILE, log_message)
    except IOError as e:
        print(f"{log_entry_prefix}Error writing to log file {LOG_FILE}: {e}")
    except Exception as e:
        print(f"{log_entry_prefix}An unexpected error occurred during logging: {e}")

    return j


def calculate_service_value(total_input_tokens, total_output_tokens):
    """Calculate service value based on input and output tokens"""
    return total_input_tokens + 2 * total_output_tokens


async def fairness_result(clients, exp_type, logger):
    # Calculate service values and max service in one pass
    max_service = 0
    service = []
    raw_throughputs = []
    raw_latencies = []
    raw_costs = []

    logger.debug(f"[Fairness Debug] Calculating fairness for {len(clients)} clients")

    for client in clients:
        # Get latest results
        latest_result = client.results[-1]

        # Calculate service value
        service_value = calculate_service_value(
            latest_result["total_input_tokens"],
            latest_result["total_output_tokens"]
        )

        if "QUE" in exp_type:
            raw_throughputs.append(latest_result['tokens_per_second']['average'])
            raw_latencies.append(latest_result['latency']['average'])
            raw_costs.append(service_value)

        logger.debug(f"[Fairness Debug] Client {latest_result['client_index']}: "
                     f"input_tokens={latest_result['total_input_tokens']}, "
                     f"output_tokens={latest_result['total_output_tokens']}, "
                     f"service_value={service_value}")

        # Update max service
        max_service = max(max_service, service_value)

        # Store service info
        service.append({
            "service": service_value,
            "client": latest_result["client_index"]
        })

        # Update client attributes
        client.service = service_value

    if "QUE" in exp_type:
        throughput_min, throughput_max = min(raw_throughputs), max(raw_throughputs)
        latency_min, latency_max = min(raw_latencies), max(raw_latencies)
        cost_min, cost_max = min(raw_costs), max(raw_costs)

    logger.debug(f"[Fairness Debug] Max service calculated: {max_service}")

    # 添加除零保护：如果max_service为0，说明所有客户端都没有处理任何token
    if max_service == 0:
        logger.warning(
            f"[Fairness] Warning: max_service is 0, all clients have zero tokens. Setting equal fairness ratios.")
        # 如果没有服务值，给所有客户端相等的公平性比例
        for client in clients:
            client.fairness_ratio = 1  # 平均分配

        # 计算Jain's公平性指数
        tmp_jains_index = calculate_Jains_index(clients, exp_type)
        return tmp_jains_index, service

    # Calculate fairness ratios in one pass
    alpha = GLOBAL_CONFIG['alpha']
    for client in clients:
        slo_violation_ratio = client.slo_violation_count / client.results[-1]['total_requests']
        service_ratio = client.service / max_service  # 现在max_service保证不为0
        client.fairness_ratio = service_ratio * (1 - alpha) + alpha * slo_violation_ratio

        if "QUE" in exp_type:
            current_throughput = client.results[-1]['tokens_per_second']['average']
            current_latency = client.results[-1]['latency']['average']
            current_cost = client.service

            # --- 开始计算归一化值 ---

            # a. 吞吐量 (越高越好)
            # 公式: (current - min) / (max - min)
            if (throughput_max - throughput_min) > 0:
                client.Norm_throughput = (current_throughput - throughput_min) / (throughput_max - throughput_min)
            else:
                client.Norm_throughput = 0.5  # 如果所有值都一样，给一个中间值

            # b. 延迟 (越低越好)
            # 反向归一化公式: (max - current) / (max - min)
            if (latency_max - latency_min) > 0:
                client.Norm_latency = (latency_max - current_latency) / (latency_max - latency_min)
            else:
                client.Norm_latency = 0.5

            # c. 成本 (越低越好)
            # 反向归一化公式: (max - current) / (max - min)
            if (cost_max - cost_min) > 0:
                client.Norm_cost = (cost_max - current_cost) / (cost_max - cost_min)
            else:
                client.Norm_cost = 0.5
            client.que = GLOBAL_CONFIG.get('que_throughput', 0.3) * client.Norm_throughput - GLOBAL_CONFIG.get(
                'que_latency', 0.4) * client.Norm_latency - GLOBAL_CONFIG.get('que_cost', 0.3) * client.Norm_cost

    # Calculate Jain's fairness index
    tmp_jains_index = calculate_Jains_index(clients, exp_type)

    return tmp_jains_index, service


async def is_fairness_LFSLLM(clients, exp_type):
    if len(clients) < 2:
        print("[Fairness] Not enough clients for fairness calculation (minimum 2 required)")
        return
    iteration = 0
    count = 0

    while iteration < (len(clients) / 2):
        print(f"[Fairness] Starting iteration {iteration + 1}/{len(clients) / 2}")
        clients.sort(key=lambda client: client.fairness_ratio)
        client_low_fairness_ratio, client_high_fairness_ratio = selectClients_LFS(clients)
        if client_low_fairness_ratio is not None and client_high_fairness_ratio is not None:
            exchange_resources(client_low_fairness_ratio, client_high_fairness_ratio, clients, exp_type)
            count += 1
        else:
            return count
        iteration += 1

    print("[Fairness] WARNING: Reached maximum iterations without achieving target fairness ratio")
    return count


async def is_fairness_QUE(clients, exp_type):
    if len(clients) < 2:
        print("[Fairness] Not enough clients for fairness calculation (minimum 2 required)")
        return

    # --- 核心逻辑开始 ---

    # 1. 计算所有客户端的平均que
    que_scores = [client.que for client in clients]
    average_que = sum(que_scores) / len(que_scores)

    # 2. 选出que最小的客户端
    worst_client = min(clients, key=lambda c: c.que)

    # 3. 计算需要增加的优先级
    # 公式: (que平均值 - client.que) * 10
    # 由于 worst_client.que 是最小值，(average_que - worst_client.que) 必然是正数
    priority_boost = (average_que - worst_client.que) * 10

    print(f"平均que为: {average_que:.2f}")
    print(f"找到体验最差的客户端: {worst_client.name} (que: {worst_client.que:.2f})")
    print(f"计算出的优先级提升量: ({average_que:.2f} - {worst_client.que:.2f}) * 10 = {priority_boost:.2f}")

    # 4. 更新该客户端的优先级
    original_priority = worst_client.priority
    worst_client.priority += priority_boost

    print(f"客户端 {worst_client.name} 的优先级从 {original_priority:.2f} 提升至 {worst_client.priority:.2f}")

    return 0


async def is_fairness_FCFS(clients, exp_type):
    # if len(clients) < 2:
    #     print("[Fairness] Not enough clients for fairness calculation (minimum 2 required)")
    #     return
    # iteration = 0
    # count = 0
    # while iteration < (len(clients) / 2):
    #     print(f"[Fairness] Starting iteration {iteration + 1}/{len(clients) / 2}")
    #     clients.sort(key=lambda client: client.service)
    #     client1, client2 = selectClients_VTC(clients)
    #     if client1 is not None and client2 is not None:
    #         exchange_resources(client1, client2, clients, exp_type)
    #         count += 1
    #     else:
    #         break
    #     iteration += 1

    return 0


def calculate_percentile(values, percentile, reverse=False):
    """Calculate percentile value from a list"""
    if not values:
        return None
    target_percentile = 100 - percentile if reverse else percentile
    return np.percentile(values, target_percentile)


def calculate_metrics(concurrency, request_timeout, client_id, results, start_time, end_time, num_requests, qps,
                      output_tokens, latency_slo, fairness_ratio, drift_time, credit, timeout_count):
    # 添加调试信息
    print(f"[Debug] calculate_metrics for {client_id}: {len(results)} results")
    if len(results) > 0:
        print(f"[Debug] First few results: {results[:3]}")

    # Calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _, _, _ in results if tokens is not None)
    total_input_tokens = sum(input_token for _, _, _, _, input_token, _ in results if input_token is not None)

    # 添加token调试信息
    print(f"[Debug] {client_id}: total_output_tokens={total_tokens}, total_input_tokens={total_input_tokens}")

    latencies = [elapsed_time for _, elapsed_time, _, _, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _, _, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft, _, _ in results if ttft is not None]
    slo_violation_count = len([slo for _, _, _, _, _, slo in results if slo == 0])
    avg_latency_div_standard_latency = sum(latencies) / len(latencies) / (latency_slo if latency_slo > 0 else 0)

    successful_requests = len(results)
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(
        tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0

    # Calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]

    return {
        "credit": credit,
        "drift_time": drift_time,
        "latency_slo": latency_slo,
        "slo_violation_count": slo_violation_count + timeout_count,
        "avg_latency_div_standard_latency": avg_latency_div_standard_latency,
        "time": end_time,
        "qps": qps,
        "fairness_ratio": fairness_ratio,
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "latency": {
            "average": avg_latency,
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        },
        "client_index": client_id,
    }
