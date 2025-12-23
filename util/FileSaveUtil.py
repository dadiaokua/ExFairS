import json
import os
import time
import logging


def save_results(exchange_count, f_result, s_result, RESULTS_FILE, logger=None):
    """将公平性结果追加写入 JSON 文件
    
    Args:
        exchange_count: Number of resource exchanges
        f_result: Can be either a float (old format) or dict with keys 'safi', 'token', 'slo_violation'
        s_result: Service result
        RESULTS_FILE: Path to results file
        logger: Optional logger
    """
    # 获取当前时间
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Handle both old format (float) and new format (dict)
    if isinstance(f_result, dict):
        new_entry = {
            "jains_index_safi": f_result.get("safi", 0),
            "jains_index_token": f_result.get("token", 0),
            "jains_index_slo_violation": f_result.get("slo_violation", 0),
            "f_result": f_result.get("safi", 0),  # Keep for backward compatibility
            "s_result": s_result,
            "time": formatted_time,
            "exchange_count": exchange_count
        }
    else:
        # Old format compatibility
        new_entry = {
            "f_result": f_result,
            "s_result": s_result,
            "time": formatted_time,
            "exchange_count": exchange_count
        }

    # 读取原有内容并追加
    if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
        try:
            with open(RESULTS_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    if logger:
                        logger.warning(f"Content of {RESULTS_FILE} is not a list. Initializing a new list.")
                    else:
                        print(f"Warning: Content of {RESULTS_FILE} is not a list. Initializing a new list.")
                    data = []
        except Exception as e:
            if logger:
                logger.warning(f"Could not decode JSON from {RESULTS_FILE}: {e}. Initializing a new list.")
            else:
                print(f"Warning: Could not decode JSON from {RESULTS_FILE}: {e}. Initializing a new list.")
            data = []
    else:
        data = []

    data.append(new_entry)
    save_json(data, RESULTS_FILE, logger=logger)


def save_json(data, filepath, mode='w', indent=2, logger=None):
    """
    通用的JSON保存函数。
    - data: 要保存的数据
    - filepath: 文件路径
    - mode: 写入模式，'w'覆盖，'a'追加
    - indent: JSON缩进
    - logger: 可选日志记录器
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        if mode == 'a' and os.path.exists(filepath):
            # 追加模式下，先读取原有内容
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
            if isinstance(existing, list) and isinstance(data, list):
                data = existing + data
            elif isinstance(existing, list):
                data = existing + [data]
            else:
                # 其他情况直接覆盖
                pass
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        if logger:
            logger.info(f"Saved data to {filepath}")
        else:
            print(f"Saved data to {filepath}")
    except Exception as e:
        msg = f"Error saving data to {filepath}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)


def save_benchmark_results(filename, benchmark_results, plot_data, logger=None):
    """保存基准测试结果和绘图数据"""
    save_json(benchmark_results, os.path.join("results", filename), logger=logger)
    save_json(plot_data, "tmp_result/plot_data.json", logger=logger)


def save_exchange_record(record, filepath, logger=None):
    """保存交换记录到文件（以列表形式追加）"""
    # 读取原有内容并追加
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                records = json.load(f)
            except Exception:
                records = []
    else:
        records = []
    records.append(record)
    save_json(records, filepath, logger=logger)


def save_to_file(filename, data, logger=None):
    """以文本方式追加保存"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(data + "\n")
        if logger:
            logger.info(f"Appended data to {filename}")
        else:
            print(f"Appended data to {filename}")
    except Exception as e:
        msg = f"Error appending data to {filename}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
