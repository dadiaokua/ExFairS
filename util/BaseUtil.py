import json
from datetime import datetime
from openai import AsyncOpenAI

from config.Config import GLOBAL_CONFIG
from util.FileSaveUtil import save_exchange_record


def initialize_clients(local_port):
    """Initialize OpenAI clients based on port configuration"""
    try:
        if isinstance(local_port, list):
            print(f"Initializing multiple OpenAI clients:")
            clients = []
            for port in local_port:
                url = f"http://localhost:{port}/v1"
                print(f"Creating client with base_url: {url}")
                client = AsyncOpenAI(base_url=url, api_key="empty")
                if client is None:
                    raise ValueError(f"Failed to create OpenAI client for port {port}")
                clients.append(client)
            print(f"✓ Successfully created {len(clients)} OpenAI clients")
            return clients
        else:
            url = f"http://localhost:{local_port}/v1"
            print(f"Initializing single OpenAI client with base_url: {url}")
            client = AsyncOpenAI(base_url=url, api_key="empty")
            if client is None:
                raise ValueError(f"Failed to create OpenAI client for port {local_port}")
            print(f"✓ Successfully created 1 OpenAI client")
            return [client]
    except Exception as e:
        print(f"❌ Error initializing OpenAI clients: {e}")
        raise


def exchange_resources(client_low_fairness_ratio, client_high_fairness_ratio, clients, exp_type):
    """
    在两个客户端之间交换资源以提高公平性
    
    Args:
        client_low_fairness_ratio: fairness_ratio 较低的客户端
        client_high_fairness_ratio: fairness_ratio 较高的客户端
        clients: 所有客户端列表，用于计算平均成功率
        exp_type: 实验类型
    """
    # 1. 计算调整量
    if "LFS" in exp_type:
        delta = calculate_adjustment_delta_lfs(client_low_fairness_ratio, client_high_fairness_ratio)
    elif "VTC" in exp_type or "FCFS" in exp_type:
        delta = calculate_adjustment_delta_vtc(client_low_fairness_ratio, client_high_fairness_ratio)
    else:
        print(f"Invalid experiment type: {exp_type}")
        return

    # 3. 获取系统平均成功率
    avg_success_rate = get_average_success_rate(clients)
    if avg_success_rate is None:
        return  # 没有足够数据进行调整

    # 4. 根据系统负载调整资源
    adjust_resources(client_low_fairness_ratio, client_high_fairness_ratio, delta, avg_success_rate)

    # 5. 更新信用值和交换次数
    update_credits_and_counters(client_low_fairness_ratio, client_high_fairness_ratio, delta)

    # 准备客户端信息列表
    clients_info = []
    for client in clients:
        client_info = {
            "client_id": client.client_id if hasattr(client, 'client_id') else str(client),
            "fairness_ratio": client.fairness_ratio,
            "service": client.service,
            "credit": client.credit,
            "latency_slo": client.latency_slo if hasattr(client, 'latency_slo') else 0,
            "exchange_Resources_Times": client.exchange_Resources_Times
        }
        clients_info.append(client_info)

    # 准备交换记录
    exchange_record = {
        "round": len(client_low_fairness_ratio.results),
        "timestamp": datetime.now().strftime("%m_%d_%H_%M_%S"),
        "client1_id": client_low_fairness_ratio.client_id if hasattr(client_low_fairness_ratio, 'client_id') else str(
            client_low_fairness_ratio),
        "client2_id": client_high_fairness_ratio.client_id if hasattr(client_high_fairness_ratio, 'client_id') else str(
            client_high_fairness_ratio),
        "gap_fairness_ratio": f"abs({client_low_fairness_ratio.fairness_ratio} - {client_high_fairness_ratio.fairness_ratio}) = {abs(client_low_fairness_ratio.fairness_ratio - client_high_fairness_ratio.fairness_ratio)}",
        "delta": delta,
        "client1_new_time_ratio": client_low_fairness_ratio.time_ratio,
        "client1_new_credit": client_low_fairness_ratio.credit,
        "client2_new_credit": client_high_fairness_ratio.credit,
        "clients_info": clients_info
    }

    save_exchange_record(exchange_record,
                         f'tmp_result/{exp_type}_resources_exchanges_{GLOBAL_CONFIG["monitor_file_time"]}.json')


def calculate_adjustment_delta_lfs(client1, client2):
    """计算调整量"""
    fairness_diff = abs(client1.fairness_ratio - client2.fairness_ratio)
    delta = fairness_diff * GLOBAL_CONFIG["ADJUST_SENSITIVITY"]
    max_delta = GLOBAL_CONFIG.get("MAX_ADJUST_DELTA", 0.5)
    return min(delta, max_delta)


def calculate_adjustment_delta_vtc(client1, client2):
    """计算调整量"""
    fairness_diff = abs(client1.service - client2.service) / max(client1.service, client2.service)
    delta = fairness_diff * GLOBAL_CONFIG.get("ADJUST_SENSITIVITY", 1.5)
    max_delta = GLOBAL_CONFIG.get("MAX_ADJUST_DELTA", 0.5)
    return min(delta, max_delta)


def get_average_success_rate(clients):
    """计算系统平均成功率"""
    success_rates = []
    for client in clients:
        if client.results:
            rate = client.results[-1]['successful_requests'] / client.results[-1]['total_requests']
            success_rates.append(rate)

    if not success_rates:
        print("No clients have successful requests")
        return None

    avg_rate = sum(success_rates) / len(success_rates)
    print(f"System average success rate: {avg_rate:.2f}")
    return avg_rate


def clip_priority(priority):
    """
    【改进1】优先级边界限制
    将优先级裁剪到 [priority_min, priority_max] 范围内，防止极端值
    """
    priority_min = GLOBAL_CONFIG.get("priority_min", -100)
    priority_max = GLOBAL_CONFIG.get("priority_max", 100)
    return max(priority_min, min(priority_max, priority))


def decay_priorities(clients):
    """
    【改进2】优先级衰减/回正
    每轮结束后让所有客户端的优先级向 0 收敛，避免历史粘滞
    
    衰减公式：new_priority = old_priority * decay_rate
    例如：decay_rate=0.9 时，优先级 100 → 90 → 81 → 73 → ...
    
    Args:
        clients: 客户端列表
    """
    decay_rate = GLOBAL_CONFIG.get("priority_decay_rate", 0.95)
    
    decayed_clients = []
    for client in clients:
        if hasattr(client, 'priority') and client.priority != 0:
            old_priority = client.priority
            # 向 0 衰减
            new_priority = int(old_priority * decay_rate)
            # 如果衰减后绝对值小于 1，直接归零
            if abs(new_priority) < 1:
                new_priority = 0
            client.priority = new_priority
            if old_priority != new_priority:
                decayed_clients.append(f"{client.client_id}: {old_priority} -> {new_priority}")
    
    if decayed_clients:
        print(f"[ExFairS] Priority decay (rate={decay_rate}): {', '.join(decayed_clients)}")


def adjust_resources(client_low_fairness_ratio, client_high_fairness_ratio, delta, avg_success_rate):
    """
    统一的资源调整策略 - ExFairS 核心
    
    调整逻辑：
    - fairness_ratio 高的客户端 = 体验差（SLO违约多或资源少）→ 需要提升优先级
    - fairness_ratio 低的客户端 = 体验好 → 降低优先级，让出资源
    
    优先级规则：数字越小优先级越高（负数 > 0 > 正数）
    
    改进：
    - 【改进1】优先级边界限制：防止极端插队
    - 【改进3】小步调整：使用可配置的放大系数，减缓累积速度
    """
    # 【改进3】使用可配置的放大系数（默认15，原值20）
    priority_amplifier = GLOBAL_CONFIG.get("priority_amplifier", 15)
    priority_changes = max(1, int(delta * priority_amplifier))  # 至少变化 1
    
    # 根据 SLO 违约情况进一步调整
    # 如果 high_fairness_ratio 客户端的 SLO 违约率很高，额外提升其优先级
    if hasattr(client_high_fairness_ratio, 'slo_violation_count') and hasattr(client_high_fairness_ratio, 'results'):
        if client_high_fairness_ratio.results:
            total_req = client_high_fairness_ratio.results[-1].get('total_requests', 1)
            violation_rate = client_high_fairness_ratio.slo_violation_count / total_req
            if violation_rate > 0.5:  # 违约率超过 50%
                priority_changes = int(priority_changes * 1.5)  # 额外提升 50%
                print(f"[ExFairS] High violation rate ({violation_rate:.2%}) detected, boosting priority change to {priority_changes}")
    
    # 公平性高的客户端（体验差）→ 获得更高优先级（更小的数字）
    old_high_priority = client_high_fairness_ratio.priority
    new_high_priority = client_high_fairness_ratio.priority - priority_changes
    # 【改进1】应用优先级边界限制
    client_high_fairness_ratio.priority = clip_priority(new_high_priority)
    
    # 公平性低的客户端（体验好）→ 获得更低优先级（更大的数字）
    old_low_priority = client_low_fairness_ratio.priority
    new_low_priority = client_low_fairness_ratio.priority + priority_changes
    # 【改进1】应用优先级边界限制
    client_low_fairness_ratio.priority = clip_priority(new_low_priority)
    
    # 打印调整信息（包含是否被裁剪的提示）
    high_clipped = " [CLIPPED]" if client_high_fairness_ratio.priority != new_high_priority else ""
    low_clipped = " [CLIPPED]" if client_low_fairness_ratio.priority != new_low_priority else ""
    
    print(f"[ExFairS] Priority adjustment: "
          f"client_high({client_high_fairness_ratio.client_id}): {old_high_priority} -> {client_high_fairness_ratio.priority}{high_clipped}, "
          f"client_low({client_low_fairness_ratio.client_id}): {old_low_priority} -> {client_low_fairness_ratio.priority}{low_clipped}")


def update_credits_and_counters(client1, client2, delta):
    """更新信用值和交换次数"""
    credit_change = int(delta * 10)
    client1.credit += credit_change
    client2.credit -= credit_change

    client1.exchange_Resources_Times += 1
    client2.exchange_Resources_Times += 1


def selectClients_LFS(clients):
    """
    从客户端列表中选择两个客户端进行资源交换
    优化版本：使用单次遍历和更高效的数据结构
    """
    n = len(clients)
    if n < 2:
        return None, None

    # 使用列表推导式快速过滤出可交换的客户端
    eligible_clients = [
        client for client in clients
        if client.exchange_Resources_Times < GLOBAL_CONFIG.get('max_exchange_times', 3)
    ]

    if len(eligible_clients) < 2:
        return None, None

    # 按fairness_ratio排序
    eligible_clients.sort(key=lambda x: x.fairness_ratio)

    # 使用双指针快速找到最大差距对
    max_diff = 0
    best_pair = None

    # 从两端向中间移动，找到最大差距对
    left, right = 0, len(eligible_clients) - 1
    while left < right:
        diff = abs(eligible_clients[left].fairness_ratio - eligible_clients[right].fairness_ratio)
        if diff > GLOBAL_CONFIG['fairness_ratio_exfairs']:
            if diff > max_diff:
                max_diff = diff
                best_pair = (left, right)
            # 移动差距较小的一端
            if (eligible_clients[left + 1].fairness_ratio - eligible_clients[left].fairness_ratio <
                    eligible_clients[right].fairness_ratio - eligible_clients[right - 1].fairness_ratio):
                left += 1
            else:
                right -= 1
        else:
            break

    if not best_pair:
        print("[Fairness] No client pairs found with sufficient fairness ratio difference")
        return None, None

    # 在找到的范围内选择最优客户端
    low_idx, high_idx = best_pair

    # 使用min和key函数快速找到最优客户端
    best_low = min(
        eligible_clients[:low_idx + 1],
        key=lambda x: (x.exchange_Resources_Times, x.credit)
    )

    best_high = min(
        eligible_clients[high_idx:],
        key=lambda x: (x.exchange_Resources_Times, -x.credit)
    )

    print(
        f"[Fairness] Selected clients with fairness ratios: {best_low.fairness_ratio:.3f} and {best_high.fairness_ratio:.3f}")
    print(f"[Fairness] Exchange times: {best_low.exchange_Resources_Times} and {best_high.exchange_Resources_Times}")
    print(f"[Fairness] Credits: {best_low.credit} and {best_high.credit}")

    return best_low, best_high


def selectClients_VTC(clients):
    i, j = 0, len(clients) - 1

    while i < j:
        if clients[i].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            i += 1
        if clients[j].exchange_Resources_Times >= GLOBAL_CONFIG.get('max_exchange_times', 3):
            j -= 1

        if abs(clients[i].service / clients[j].service) <= GLOBAL_CONFIG.get('fairness_ratio_VTC', 0.5):
            return clients[j], clients[i]
        else:
            return None, None

    return None, None
