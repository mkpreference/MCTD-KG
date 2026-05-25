#!/bin/bash
# BGE-M3 模型下载脚本
# 用法: bash download_bge_m3.sh [目标路径]
# 默认路径: /root/files_lpz/bge-m3

set -e

MODEL_DIR="${1:-/root/files_lpz/bge-m3}"

if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "模型已存在于 $MODEL_DIR，跳过下载。"
    exit 0
fi

echo "正在下载 BGE-M3 模型到 $MODEL_DIR ..."

# 方式一：huggingface-cli（推荐，国内可用镜像）
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download BAAI/bge-m3 --local-dir "$MODEL_DIR"
# 方式二：Python sentence-transformers 下载后保存
else
    python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
model.save('$MODEL_DIR')
print('下载完成')
"
fi

echo "BGE-M3 模型已安装至 $MODEL_DIR"
