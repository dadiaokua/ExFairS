# 常量定义
# 避免魔法数字，提高代码可维护性

# Worker 相关
WORKER_EMPTY_CYCLE_THRESHOLD = 1200  # 120秒无请求后记录警告
WORKER_SLEEP_INTERVAL = 0.2  # Worker 每轮睡眠时间（秒）
REQUEST_TIMEOUT_SECONDS = 20.0  # 请求超时时间（秒）

# 批量处理
MAX_BATCH_SIZE = 16  # 每批最多处理的请求数（较小值可放大调度策略差异）

# 并发控制
MAX_CONCURRENCY = 20  # 最大并发请求数（与 vLLM max_num_seqs 对齐）

# SLO-Greedy 冷启动
COLD_START_THRESHOLD = 10  # 请求数少于此值视为新客户端
COLD_START_WEIGHT = 0.5  # 新客户端的初始违约率权重

# Justitia 策略
DEFAULT_TOTAL_MEMORY = 100000  # 抽象显存容量单位

# 优先级策略 - ExFairS 增强
DEFAULT_PRIORITY_INSERT_MULTIPLIER = 10  # 优先级插入倍数（增加以放大效果）
DEFAULT_MAX_PRIORITY_POSITIONS = 200  # 最大优先级插入位置（增加上限）
DEFAULT_MAX_PRIORITY_DELAY = 10  # 最大延迟秒数

# 队列监控
DEFAULT_QUEUE_MONITOR_INTERVAL = 5  # 队列监控间隔（秒）
DEFAULT_MAX_QUEUE_SIZE = 10000  # 默认最大队列大小
