# 常量定义
# 避免魔法数字，提高代码可维护性

# Worker 相关
WORKER_EMPTY_CYCLE_THRESHOLD = 1200  # 120秒无请求后记录警告
WORKER_SLEEP_INTERVAL = 1.0  # Worker 每轮睡眠时间（秒）
REQUEST_TIMEOUT_SECONDS = 60.0  # 请求超时时间（秒）

# 批量处理
MAX_BATCH_SIZE = 10  # 每批最多处理的请求数

# 并发控制
MAX_CONCURRENCY = 256  # 最大并发请求数（与 vLLM max_num_seqs 对齐）

# SLO-Greedy 冷启动
COLD_START_THRESHOLD = 10  # 请求数少于此值视为新客户端
COLD_START_WEIGHT = 0.5  # 新客户端的初始违约率权重

# Justitia 策略
DEFAULT_TOTAL_MEMORY = 100000  # 抽象显存容量单位

# 优先级策略
DEFAULT_PRIORITY_INSERT_MULTIPLIER = 1  # 优先级插入倍数
DEFAULT_MAX_PRIORITY_POSITIONS = 100  # 最大优先级插入位置
DEFAULT_MAX_PRIORITY_DELAY = 10  # 最大延迟秒数

# 队列监控
DEFAULT_QUEUE_MONITOR_INTERVAL = 5  # 队列监控间隔（秒）
DEFAULT_MAX_QUEUE_SIZE = 10000  # 默认最大队列大小
