import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np

from config.Config import GLOBAL_CONFIG
from plot.plotUtil import xLabel_time, plot_metrics_with_annotations, setup_subplot, setup_subplot_client, \
    plot_fairness_index

line_styles = ['-', '--', '-.', ':']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', '*']


def plot_averaged_results(short_results, long_results, args_concurrency, total_time, filename, exp_type):
    # 创建一个4×1的子图布局
    fig2, axs2 = plt.subplots(4, 1, figsize=(14, 20))

    # 添加标题显示参数
    fig2.suptitle(
        f"{exp_type} System Benchmark Results - Short Client: {args_concurrency}, Total Time: {total_time}, Alpha: {GLOBAL_CONFIG.get('alpha', 0.5)}",
        fontsize=16, y=0.98)

    # 合并所有结果
    all_results = short_results + long_results

    if not all_results:
        return

    avg_success_rate = []
    avg_tokens_per_second = []
    avg_latency = []
    avg_ttft = []
    avg_rps = []
    avg_tokens_count = []
    avg_total_output_tokens = []
    avg_total_input_tokens = []
    concurrency = []
    qps = []
    time = []

    # Get the shortest length among all results
    min_len = min(len(result) for result in all_results)

    for i in range(min_len):
        time.append(all_results[0][i]["time"])
        # 总QPS是所有客户端QPS之和
        qps.append(all_results[0][i]["qps"] * len(all_results))
        concurrency.append(all_results[0][i]["concurrency"])

        # 计算总和而不是平均值
        tokens_count = sum(result[i]['total_output_tokens'] + result[i]['total_input_tokens'] for result in all_results)
        total_output_tokens = sum(result[i]['total_output_tokens'] for result in all_results)
        total_input_tokens = sum(result[i]['total_input_tokens'] for result in all_results)
        # 成功请求数总和
        successful_requests = sum(result[i]['successful_requests'] for result in all_results)
        total_requests = sum(result[i]['total_requests'] for result in all_results)
        success_rate = successful_requests * 100 / total_requests if total_requests > 0 else 0
        # 取所有客户端延迟的和作为系统延迟
        tokens_per_second = sum(result[i]["tokens_per_second"]["p99"] for result in all_results)
        latency = max(result[i]["latency"]["p99"] for result in all_results)
        ttft = max(result[i]["time_to_first_token"]["p99"] for result in all_results)
        # 总RPS是所有客户端RPS之和
        rps = sum(result[i]["requests_per_second"] for result in all_results)

        avg_tokens_count.append(tokens_count)
        avg_total_output_tokens.append(total_output_tokens)
        avg_total_input_tokens.append(total_input_tokens)
        avg_success_rate.append(success_rate)
        avg_tokens_per_second.append(tokens_per_second)
        avg_latency.append(latency)
        avg_ttft.append(ttft)
        avg_rps.append(rps)

    time_xLabel = xLabel_time(time)

    # Plot tokens count with different line styles
    plot_metrics_with_annotations(axs2[0], range(len(qps)), avg_tokens_count, 'tokens count',
                                  colors[0], markers[0], linestyle=line_styles[0])
    plot_metrics_with_annotations(axs2[0], range(len(qps)), avg_total_output_tokens, 'total_output_tokens',
                                  colors[1], markers[1], linestyle=line_styles[1])
    plot_metrics_with_annotations(axs2[0], range(len(qps)), avg_total_input_tokens, 'total_input_tokens',
                                  colors[2], markers[2], linestyle=line_styles[2])
    setup_subplot(axs2[0],
                  "Sum Input and Output Tokens",
                  time_xLabel,
                  plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Plot success rate
    plot_metrics_with_annotations(axs2[1], range(len(qps)), avg_success_rate, 'success_rate',
                                  colors[3], markers[3], linestyle=line_styles[0])
    setup_subplot(axs2[1],
                  "Average Success Rate",
                  time_xLabel,
                  ylim=(max(min(avg_success_rate) - 10, 0), 105))

    # Plot tokens per second
    plot_metrics_with_annotations(axs2[2], range(len(qps)), avg_tokens_per_second, 'tokens_per_second',
                                  colors[4], markers[4], xytext=(0, -15), linestyle=line_styles[0])
    setup_subplot(axs2[2],
                  "Sum Tokens/s",
                  time_xLabel)

    # Plot latency metrics
    plot_metrics_with_annotations(axs2[3], range(len(qps)), avg_latency, 'latency',
                                  colors[0], markers[0], linestyle=line_styles[0])
    plot_metrics_with_annotations(axs2[3], range(len(qps)), avg_ttft, 'time_to_first_token',
                                  colors[1], markers[1], xytext=(0, -15), linestyle=line_styles[1])
    plot_metrics_with_annotations(axs2[3], range(len(qps)), avg_rps, 'requests_per_second',
                                  colors[2], markers[2], xytext=(15, 0), linestyle=line_styles[2])
    setup_subplot(axs2[3],
                  "Max Latency and TTFT, Sum RPS Metrics",
                  time_xLabel)

    # 优化布局
    fig2.tight_layout(pad=3.0, h_pad=2.0)
    # 调整子图间距，为底部图例留出空间
    fig2.subplots_adjust(top=0.92, bottom=0.1)

    # 创建 figure 文件夹（如果不存在）
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 保存图片
    fig2.savefig('figure/system_results' + filename.split('.')[0] + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return time_xLabel


def plot_comprehensive_results(sorted_all_results, args_concurrency, total_time, filename, exp_type, qps_with_time):
    """
    绘制综合性能图表，只包括各客户端的性能指标
    """
    # 创建性能指标图
    fig1, axs1 = plt.subplots(4, 2, figsize=(28, 25))  # 性能指标图
    axs1 = axs1.flatten()  # 将2D数组展平为1D，便于索引

    # 添加标题显示参数
    fig1.suptitle(
        f"{exp_type} Performance Metrics - Concurrency: {args_concurrency}, Total Time: {total_time}, Alpha: {GLOBAL_CONFIG.get('alpha', 0.5)}",
        fontsize=16, y=0.98)

    # 获取所有客户端
    clients = set()
    for result in sorted_all_results:
        clients.add(f'{result[0]["client_index"]}_slo_{result[0]["latency_slo"]}')

    # 将clients分为short、long和mix三组
    short_clients = sorted([c for c in clients if "short" in c])
    long_clients = sorted([c for c in clients if "long" in c])
    mix_clients = sorted([c for c in clients if "mix" in c])

    print(f"Short clients: {short_clients}")
    print(f"Long clients: {long_clients}")
    print(f"MIX clients: {mix_clients}")

    # 暖色系用于short clients
    warm_colors = ['#FF4D4D', '#FFA64D', '#FFD700', '#FF69B4', '#FF8C69']
    # 冷色系用于long clients  
    cool_colors = ['#4169E1', '#00CED1', '#6495ED', '#483D8B', '#008B8B']
    # 中性色系用于mix clients
    mix_colors = ['#9370DB', '#32CD32', '#FF6347', '#20B2AA', '#DAA520']

    # 创建一个共享图例的句柄和标签列表
    legend_handles = []
    legend_labels = []

    # 绘制性能指标
    # 1. 成功率
    plot_client_metric(axs1[0], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "success_rate",
                       "Success Rate (%)", legend_handles, legend_labels)

    # 2. 每秒令牌数
    plot_client_metric(axs1[1], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "tokens_per_second",
                       "Tokens/s", legend_handles, legend_labels)

    # 3. 延迟
    plot_client_metric(axs1[2], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "latency",
                       "Latency (ms)", legend_handles, legend_labels)

    # 4. 首个令牌时间
    plot_client_metric(axs1[3], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "time_to_first_token",
                       "Time to First Token (ms)", legend_handles, legend_labels)

    # 5. 每秒请求数
    plot_client_metric(axs1[4], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "requests_per_second",
                       "Requests per Second", legend_handles, legend_labels)

    # 6. 每个客户端的服务使用量
    plot_client_metric(axs1[5], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "service",
                       "Service Usage", legend_handles, legend_labels)

    # 7. 成功请求数
    plot_client_metric(axs1[6], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "successful_requests",
                       "Successful Requests", legend_handles, legend_labels)

    # 8. 违反SLO的请求数量
    plot_client_metric(axs1[7], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "slo_violation_count",
                       "Slo Violation Request Numbers", legend_handles, legend_labels)

    # 在性能指标图底部添加共享图例
    fig1.legend(handles=legend_handles, labels=legend_labels,
               loc='center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(short_clients) + len(long_clients) + len(mix_clients),
               fontsize=10)

    # 优化布局
    fig1.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    # 调整子图间距，为底部图例留出空间
    fig1.subplots_adjust(top=0.92, bottom=0.1)

    # 创建figure文件夹（如果不存在）
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 保存图表
    fig1.savefig(f'figure/performance_metrics{filename.split(".")[0]}.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_fairness_results(sorted_all_results, args_concurrency, total_time, filename, exp_type, fairness_results):
    """
    绘制公平性相关图表
    """
    # 创建公平性指标图
    fig2, axs2 = plt.subplots(3, 1, figsize=(28, 18))   # 公平性指标图，3行1列
    axs2 = axs2.flatten()

    # 添加标题显示参数
    fig2.suptitle(
        f"{exp_type} Fairness Metrics - Concurrency: {args_concurrency}, Total Time: {total_time}, Alpha: {GLOBAL_CONFIG.get('alpha', 0.5)}",
        fontsize=16, y=0.98)

    # 获取所有客户端
    clients = set()
    for result in sorted_all_results:
        clients.add(f'{result[0]["client_index"]}_slo_{result[0]["latency_slo"]}')

    # 将clients分为short、long和mix三组
    short_clients = sorted([c for c in clients if "short" in c])
    long_clients = sorted([c for c in clients if "long" in c])
    mix_clients = sorted([c for c in clients if "mix" in c])

    # 暖色系用于short clients
    warm_colors = ['#FF4D4D', '#FFA64D', '#FFD700', '#FF69B4', '#FF8C69']
    # 冷色系用于long clients  
    cool_colors = ['#4169E1', '#00CED1', '#6495ED', '#483D8B', '#008B8B']
    # 中性色系用于mix clients
    mix_colors = ['#9370DB', '#32CD32', '#FF6347', '#20B2AA', '#DAA520']

    # 创建一个共享图例的句柄和标签列表
    legend_handles = []
    legend_labels = []

    # 绘制公平性指标
    # 1. Fairness Ratio
    plot_client_metric(axs2[0], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "fairness_ratio",
                       "Fairness Ratio", legend_handles, legend_labels)

    # 2. Jain's公平性指数
    if fairness_results:
        f_values = [result['f_result'] for result in fairness_results]
        times = list(range(len(f_values)))
        if len(times) == len(f_values):
            plot_fairness_index(axs2[1], f_values, times)
        else:
            print(f"Warning: Mismatched lengths - times: {len(times)}, f_values: {len(f_values)}")
            min_len = min(len(times), len(f_values))
            plot_fairness_index(axs2[1], f_values[:min_len], times[:min_len])
    else:
        # 如果没有fairness数据，显示空图表
        axs2[1].text(0.5, 0.5, 'No Fairness Data Available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axs2[1].transAxes, fontsize=12)
        axs2[1].set_title("Jain's Fairness Index")
        axs2[1].set_xlabel("Time")
        axs2[1].set_ylabel("Fairness Index")
    
    # 3. credit值
    plot_client_metric(axs2[2], sorted_all_results, short_clients, long_clients, mix_clients,
                       warm_colors, cool_colors, mix_colors, "credit",
                       "Clients Credit", legend_handles, legend_labels, ylim=None)

    # 为第三张图添加图例
    axs2[2].legend()

    # 优化布局
    fig2.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    # 调整子图间距，为底部图例留出空间
    fig2.subplots_adjust(top=0.92, bottom=0.1)

    # 创建figure文件夹（如果不存在）
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 保存图表
    fig2.savefig(f'figure/fairness_metrics{filename.split(".")[0]}.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_client_metric(ax, sorted_all_results, short_clients, long_clients, mix_clients, warm_colors, cool_colors, mix_colors, metric_key, title,
                       legend_handles, legend_labels, ylim=None):
    plotted_value_sets = []
    """绘制客户端指标"""
    
    # 初始化默认的time_xLabel，以防没有任何客户端
    time_xLabel = ["No Data"]
    
    # 获取所有客户端
    all_clients = short_clients + long_clients + mix_clients
    
    # 如果没有任何客户端，显示空图表
    if not all_clients:
        ax.text(0.5, 0.5, 'No Client Data Available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        return
    
    # 先绘制short clients，使用实线
    for i, client in enumerate(short_clients):
        client_results = next(
            r for r in sorted_all_results if f'{r[0]["client_index"]}_slo_{r[0]["latency_slo"]}' == client)
        time = [result['time'] for result in client_results]
        time_xLabel = xLabel_time(time)

        if metric_key == "success_rate":
            values = [result['successful_requests'] * 100 / result['total_requests'] for result in client_results]
        elif metric_key in ["tokens_per_second", "latency", "time_to_first_token"]:
            values = [result[metric_key]["p99"] for result in client_results]
        elif metric_key == "service":
            values = [result['total_input_tokens'] + result['total_output_tokens'] * 2 for result in client_results]
        else:
            values = [result[metric_key] for result in client_results]

        # 检查是否有相同长度和相同值的数组，避免形状不匹配的错误
        is_duplicate = any(len(values) == len(prev) and np.allclose(values, prev, atol=1e-4) for prev in plotted_value_sets)
        if is_duplicate:
            x_vals = [j + 0.02 for j in range(len(time))]  # 偏移一点
        else:
            x_vals = list(range(len(time)))

        # 使用不同的marker和linestyle来区分不同的线条
        line = ax.plot(x_vals, values, marker=markers[i % len(markers)],
                       color=warm_colors[i % len(warm_colors)], linestyle='-',
                       label=client, alpha=0.7, markersize=8)[0]

        # 👇 保存已画过的 values
        plotted_value_sets.append(values)

        if client not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(client)

    # 再绘制long clients，使用虚线
    for i, client in enumerate(long_clients):
        client_results = next(
            r for r in sorted_all_results if f'{r[0]["client_index"]}_slo_{r[0]["latency_slo"]}' == client)
        time = [result['time'] for result in client_results]
        time_xLabel = xLabel_time(time)

        if metric_key == "success_rate":
            values = [result['successful_requests'] * 100 / result['total_requests'] for result in client_results]
        elif metric_key in ["tokens_per_second", "latency", "time_to_first_token"]:
            values = [result[metric_key]["p99"] for result in client_results]
        elif metric_key == "service":
            values = [result['total_input_tokens'] + result['total_output_tokens'] * 2 for result in client_results]
        else:
            values = [result[metric_key] for result in client_results]

        # 检查是否有相同长度和相同值的数组，避免形状不匹配的错误
        is_duplicate = any(len(values) == len(prev) and np.allclose(values, prev, atol=1e-4) for prev in plotted_value_sets)
        if is_duplicate:
            x_vals = [j + 0.02 for j in range(len(time))]  # 偏移一点
        else:
            x_vals = list(range(len(time)))

        # 使用不同的marker和linestyle来区分不同的线条
        line = ax.plot(x_vals, values, marker=markers[(i + len(short_clients)) % len(markers)],
                       color=cool_colors[i % len(cool_colors)], linestyle='--',
                       label=client, alpha=0.7, markersize=8)[0]

        plotted_value_sets.append(values)

        if client not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(client)

    # 再绘制mix clients，使用点线
    for i, client in enumerate(mix_clients):
        client_results = next(
            r for r in sorted_all_results if f'{r[0]["client_index"]}_slo_{r[0]["latency_slo"]}' == client)
        time = [result['time'] for result in client_results]
        time_xLabel = xLabel_time(time)

        if metric_key == "success_rate":
            values = [result['successful_requests'] * 100 / result['total_requests'] for result in client_results]
        elif metric_key in ["tokens_per_second", "latency", "time_to_first_token"]:
            values = [result[metric_key]["p99"] for result in client_results]
        elif metric_key == "service":
            values = [result['total_input_tokens'] + result['total_output_tokens'] * 2 for result in client_results]
        else:
            values = [result[metric_key] for result in client_results]

        # 检查是否有相同长度和相同值的数组，避免形状不匹配的错误
        is_duplicate = any(len(values) == len(prev) and np.allclose(values, prev, atol=1e-4) for prev in plotted_value_sets)
        if is_duplicate:
            x_vals = [j + 0.02 for j in range(len(time))]  # 偏移一点
        else:
            x_vals = list(range(len(time)))

        # 使用不同的marker和linestyle来区分不同的线条
        line = ax.plot(x_vals, values, marker=markers[(i + len(short_clients) + len(long_clients)) % len(markers)],
                       color=mix_colors[i % len(mix_colors)], linestyle=':',
                       label=client, alpha=0.7, markersize=8)[0]

        plotted_value_sets.append(values)

        if client not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(client)

    # 计算所有值以确定合适的y轴范围
    all_values = []
    for client in all_clients:
        try:
            client_results = next(
                r for r in sorted_all_results if f'{r[0]["client_index"]}_slo_{r[0]["latency_slo"]}' == client)
            if metric_key == "success_rate":
                values = [result['successful_requests'] * 100 / result['total_requests'] for result in client_results]
            elif metric_key in ["tokens_per_second", "latency", "time_to_first_token"]:
                values = [result[metric_key]["p99"] for result in client_results]
            elif metric_key == "service":
                values = [result['total_input_tokens'] + result['total_output_tokens'] * 2 for result in client_results]
            else:
                values = [result[metric_key] for result in client_results]
            all_values.extend(values)
        except (StopIteration, KeyError, ZeroDivisionError) as e:
            print(f"Warning: Error processing {client} for {metric_key}: {e}")

    print(f"[DEBUG] {metric_key} all_values: {all_values}")

    # 如果没有提供ylim且有足够的值，则自动计算合适的范围
    if ylim is None and all_values:
        try:
            # 过滤掉None值
            valid_values = [v for v in all_values if v is not None]
            if valid_values:  # 确保有有效值
                min_val = min(valid_values)
                max_val = max(valid_values)
                # 增加最小margin，避免y轴范围过小
                margin = max((max_val - min_val) * 0.1, 1.0)
                if min_val < 0:
                    computed_ylim = (min_val - margin, max_val + margin)
                else:
                    computed_ylim = (max(0, min_val - margin), max_val + margin)
                # 特殊处理 success_rate
                if metric_key == "success_rate":
                    computed_ylim = (max(0, min_val - 5), 105)
            else:
                computed_ylim = None  # 如果没有有效值，设置为None
        except (ValueError, TypeError) as e:
            print(f"Warning: Error computing ylim for {metric_key}: {e}")
            computed_ylim = None
    else:
        computed_ylim = ylim

    # 设置子图属性
    setup_subplot_client(ax, title, time_xLabel, ylim=computed_ylim)


def plot_aggregated_results(sorted_all_results, args_concurrency, total_time, filename, exp_type):
    """
    绘制所有客户端的汇总图表（总和或平均值）
    """
    # 创建汇总图表
    fig, axs = plt.subplots(4, 2, figsize=(28, 25))
    axs = axs.flatten()

    # 添加标题
    fig.suptitle(
        f"{exp_type} Aggregated System Metrics - Concurrency: {args_concurrency}, Total Time: {total_time}, Alpha: {GLOBAL_CONFIG.get('alpha', 0.5)}",
        fontsize=16, y=0.98)

    if not sorted_all_results:
        return

    # 获取时间轴（使用第一个客户端的时间）
    time_data = [result['time'] for result in sorted_all_results[0]]
    time_xLabel = xLabel_time(time_data)
    x_values = list(range(len(time_data)))

    # 定义指标和其汇总方式
    metrics_config = [
        {
            'key': 'success_rate',
            'title': 'Average Success Rate (%)',
            'method': 'average',
            'extractor': lambda result: result['successful_requests'] * 100 / result['total_requests'] if result['total_requests'] > 0 else 0
        },
        {
            'key': 'tokens_per_second',
            'title': 'Total Tokens/s',
            'method': 'sum',
            'extractor': lambda result: result['tokens_per_second']['p99']
        },
        {
            'key': 'latency',
            'title': 'Average Latency (s)',
            'method': 'average',
            'extractor': lambda result: result['latency']['p99']
        },
        {
            'key': 'time_to_first_token',
            'title': 'Average Time to First Token (s)',
            'method': 'average',
            'extractor': lambda result: result['time_to_first_token']['p99'] / 1000
        },
        {
            'key': 'requests_per_second',
            'title': 'Total Requests per Second',
            'method': 'sum',
            'extractor': lambda result: result['requests_per_second']
        },
        {
            'key': 'service',
            'title': 'Total Service Usage',
            'method': 'sum',
            'extractor': lambda result: result['total_input_tokens'] + result['total_output_tokens'] * 2
        },
        {
            'key': 'successful_requests',
            'title': 'Total Successful Requests',
            'method': 'sum',
            'extractor': lambda result: result['successful_requests']
        },
        {
            'key': 'slo_violation_count',
            'title': 'Total SLO Violation Count',
            'method': 'sum',
            'extractor': lambda result: result['slo_violation_count']
        }
    ]

    # 为每个指标计算汇总值并绘图
    for idx, metric_config in enumerate(metrics_config):
        aggregated_values = []
        
        # 对每个时间点计算汇总值
        for time_idx in range(len(time_data)):
            time_values = []
            
            # 收集所有客户端在该时间点的值
            for client_results in sorted_all_results:
                if time_idx < len(client_results):
                    try:
                        value = metric_config['extractor'](client_results[time_idx])
                        time_values.append(value)
                    except (KeyError, ZeroDivisionError, TypeError) as e:
                        print(f"Warning: Error extracting {metric_config['key']} at time {time_idx}: {e}")
                        continue
            
            # 根据方法计算汇总值
            if time_values:
                if metric_config['method'] == 'sum':
                    aggregated_value = sum(time_values)
                elif metric_config['method'] == 'average':
                    aggregated_value = sum(time_values) / len(time_values)
                else:
                    aggregated_value = sum(time_values)  # 默认使用总和
                aggregated_values.append(aggregated_value)
            else:
                aggregated_values.append(0)

        # 绘制图表
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = line_styles[idx % len(line_styles)]
        
        axs[idx].plot(x_values, aggregated_values, 
                     marker=marker, color=color, linestyle=linestyle,
                     linewidth=2, markersize=6, alpha=0.8)
        
        # 添加数值标注（每隔几个点标注一次，避免过于拥挤）
        annotation_step = max(1, len(aggregated_values) // 8)
        for i in range(0, len(aggregated_values), annotation_step):
            axs[idx].annotate(f'{aggregated_values[i]:.1f}',
                             (x_values[i], aggregated_values[i]),
                             xytext=(0, 10), textcoords='offset points',
                             ha='center', va='bottom', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # 设置子图属性
        axs[idx].set_title(metric_config['title'], fontsize=14, fontweight='bold')
        axs[idx].set_xlabel('Time', fontsize=12)
        axs[idx].set_ylabel(metric_config['title'].split()[-1] if len(metric_config['title'].split()) > 1 else 'Value', fontsize=12)
        axs[idx].grid(True, alpha=0.3)
        axs[idx].set_xticks(range(0, len(time_xLabel), max(1, len(time_xLabel) // 10)))
        axs[idx].set_xticklabels([time_xLabel[i] for i in range(0, len(time_xLabel), max(1, len(time_xLabel) // 10))], 
                                rotation=45, ha='right')

        # 设置y轴范围
        if aggregated_values:
            min_val = min(aggregated_values)
            max_val = max(aggregated_values)
            margin = (max_val - min_val) * 0.1 if max_val != min_val else 1
            
            if metric_config['key'] == 'success_rate':
                axs[idx].set_ylim(max(0, min_val - 5), 105)
            elif min_val >= 0:
                axs[idx].set_ylim(max(0, min_val - margin), max_val + margin)
            else:
                axs[idx].set_ylim(min_val - margin, max_val + margin)

        print(f"[DEBUG] {metric_config['key']} aggregated values: {aggregated_values[:5]}...{aggregated_values[-5:] if len(aggregated_values) > 5 else ''}")

    # 优化布局
    fig.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    fig.subplots_adjust(top=0.92, bottom=0.1)

    # 创建figure文件夹（如果不存在）
    if not os.path.exists('figure'):
        os.makedirs('figure')

    # 保存图表
    fig.savefig(f'figure/aggregated_metrics{filename.split(".")[0]}.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_result(plot_data):
    with open("results/" + plot_data["filename"], 'r') as f:
        all_results = json.load(f)

    # Load fairness results and create third figure
    fairness_file_path = f"tmp_result/tmp_fairness_result_{plot_data['exp']}_{GLOBAL_CONFIG.get('monitor_file_time')}.json"
    try:
        with open(fairness_file_path, 'r') as f:
            fairness_results = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Fairness results file not found at {fairness_file_path}")
        print("Continuing without fairness data...")
        fairness_results = []

    if len(all_results) >= 1:
        # Group results by short/long/mix and sort by client_index
        short_results = []
        long_results = []
        mix_results = []

        for result in all_results:
            if "short" in result[0]["client_index"]:
                short_results.append(result)
            elif "long" in result[0]["client_index"]:
                long_results.append(result)
            elif "mix" in result[0]["client_index"]:
                mix_results.append(result)

        # Sort results by client_index
        short_results.sort(key=lambda x: x[0]["client_index"])
        long_results.sort(key=lambda x: x[0]["client_index"])
        mix_results.sort(key=lambda x: x[0]["client_index"])

        # Combine sorted results with short first, then long, then mix
        sorted_all_results = short_results + long_results + mix_results

        qps_with_time = plot_averaged_results(short_results, long_results, plot_data["concurrency"],
                                              plot_data["total_time"], plot_data["filename"],
                                              plot_data["exp"])
        plot_comprehensive_results(sorted_all_results, plot_data["concurrency"], plot_data["total_time"],
                                   plot_data["filename"], plot_data["exp"], qps_with_time)
        plot_fairness_results(sorted_all_results, plot_data["concurrency"], plot_data["total_time"],
                               plot_data["filename"], plot_data["exp"], fairness_results)
        plot_aggregated_results(sorted_all_results, plot_data["concurrency"], plot_data["total_time"],
                               plot_data["filename"], plot_data["exp"])
    else:
        print("No results found")
        return


if __name__ == "__main__":
    with open("tmp_result/plot_data.json", "r") as f:
        data = json.load(f)

    plot_result(data)
