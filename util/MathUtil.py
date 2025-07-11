import os

import numpy as np

from config.Config import GLOBAL_CONFIG
from util.BaseUtil import selectClients_LFS, selectClients_VTC, exchange_resources

import datetime

from util.FileSaveUtil import save_to_file


def calculate_Jains_index(clients, exp_type):
    """
    Calculates Jain's Fairness Index for a list of clients based on their fairness_ratio.
    Logs the calculation details with a timestamp to a file.
    """
    timestamp = datetime.datetime.now().isoformat()
    log_entry_prefix = f"[{timestamp}] "

    fairness_ratio = [client.fairness_ratio for client in clients]  # Corrected typo: clienr -> client
    n = len(fairness_ratio)

    log_message = f"{log_entry_prefix}Calculating Jain's Index for {n} clients. Fairness Ratios: {fairness_ratio}. "

    if n == 0:
        j = 0  # Avoid division by zero, define result as 0 for no clients
        log_message += f"Result: {j} (n=0)."
    else:
        sum_service = sum(fairness_ratio)
        sum_squares = sum(s ** 2 for s in fairness_ratio)
        denominator = n * sum_squares

        log_message += f"Sum(ratios): {sum_service}, Sum(ratios^2): {sum_squares}, Denominator (n * Sum(ratios^2)): {denominator}. "

        if denominator == 0:
            # Handle division by zero. If all ratios are 0, fairness could be considered perfect (1),
            # but division by zero occurs. If ratios are non-zero but sum_squares is 0 (impossible for real numbers unless n=0),
            # it's an edge case. Returning 0 avoids error, though 1 might be contextually better if all ratios are equal.
            # Let's return 0 for safety against division error.
            j = 0
            log_message += f"Result: {j} (Denominator is zero)."
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

    logger.debug(f"[Fairness Debug] Calculating fairness for {len(clients)} clients")

    for client in clients:
        # Get latest results
        latest_result = client.results[-1]

        # Calculate service value
        service_value = calculate_service_value(
            latest_result["total_input_tokens"], 
            latest_result["total_output_tokens"]
        )

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

    logger.debug(f"[Fairness Debug] Max service calculated: {max_service}")

    # 添加除零保护：如果max_service为0，说明所有客户端都没有处理任何token
    if max_service == 0:
        logger.warning(f"[Fairness] Warning: max_service is 0, all clients have zero tokens. Setting equal fairness ratios.")
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

    # Calculate Jain's fairness index
    tmp_jains_index = calculate_Jains_index(clients, exp_type)

    return tmp_jains_index, service


async def is_fairness_LFSLLM(clients, exp_type):
    if len(clients) < 2:
        print("[Fairness] Not enough clients for fairness calculation (minimum 2 required)")
        return
    iteration = 0
    count = 0

    while iteration < (len(clients)  / 2):
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


async def is_fairness_VTC(clients, exp_type):
    if len(clients) < 2:
        print("[Fairness] Not enough clients for fairness calculation (minimum 2 required)")
        return
    iteration = 0
    count = 0

    while iteration < (len(clients) / 2):
        print(f"[Fairness] Starting iteration {iteration + 1}/{len(clients) / 2}")
        clients.sort(key=lambda client: client.service)
        client1, client2 = selectClients_VTC(clients)
        if client1 is not None and client2 is not None:
            exchange_resources(client1, client2, clients, exp_type)
            count += 1
        else:
            break
        iteration += 1

    print("[Fairness] WARNING: Reached maximum iterations without achieving target fairness ratio")
    return count


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
