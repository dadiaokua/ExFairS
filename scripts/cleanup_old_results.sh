#!/bin/bash

# 清理旧的混乱结果目录
# 只保留符合新结构的目录：run_YYYYMMDD_HHMMSS/

RESULTS_DIR="results"

echo "🧹 清理旧结果目录..."
echo ""

cd "$(dirname "$0")/.." || exit 1

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "❌ results 目录不存在"
    exit 1
fi

# 统计清理前的状态
total_dirs=$(find "$RESULTS_DIR" -maxdepth 1 -type d | wc -l)
echo "📊 当前状态："
echo "   总目录数: $total_dirs"
echo ""

# 列出要删除的目录
echo "🗑️  将删除以下目录："
echo ""

deleted_count=0

cd "$RESULTS_DIR" || exit 1

# 删除所有不符合 run_YYYYMMDD_HHMMSS 格式的目录
for dir in */; do
    dir="${dir%/}"  # 移除尾部的斜杠
    
    # 检查是否符合 run_YYYYMMDD_HHMMSS 格式
    if [[ ! "$dir" =~ ^run_[0-9]{8}_[0-9]{6}$ ]]; then
        echo "  - $dir"
        rm -rf "$dir"
        deleted_count=$((deleted_count + 1))
    fi
done

echo ""
echo "✅ 清理完成！"
echo "   删除了 $deleted_count 个旧目录"
echo ""

# 显示剩余的目录
remaining_dirs=$(find . -maxdepth 1 -type d -name "run_*" | wc -l)
echo "📂 剩余有效目录数: $remaining_dirs"

if [[ $remaining_dirs -gt 0 ]]; then
    echo ""
    echo "有效目录："
    ls -1 | grep "^run_"
fi

echo ""
echo "💡 提示：运行新实验时，结果会自动保存到 run_YYYYMMDD_HHMMSS/ 目录下"

