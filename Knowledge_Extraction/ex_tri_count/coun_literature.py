import os
import json
import re

# ================= 配置区 =================
# 指向您生成 chunks 的目录
CHUNKS_DIR = "/root/files_lpz/ra/实验/literature/literature_data/3_llm_chunks"

# 🛑 降噪黑名单：包含这些词的 Chunk 会被直接踢出统计
BLACKLIST = [
    'Elsevier B.V.',
    'All rights reserved',
    'IEEE',
    'Downloaded on',
    'Authorized licensed use',
    'CRediT authorship',
    'Declaration of competing interest',
    'journal homepage',
    'https://doi.org'
]

def is_noisy_chunk(chunk):
    """核心判定器：判断当前切片是否属于无用杂音或参考文献"""
    text = chunk.strip()
    
    # 1. 拦截版权声明与元数据
    if any(bad_word in text for bad_word in BLACKLIST):
        return True
        
    # 2. 拦截纯参考文献切片
    # 特征：以 [数字] 开头，或者一段文本里有大量的 [x] 引用标号
    if re.match(r'^\[\d+\]', text) or len(re.findall(r'\[\d+\]', text)) > 6:
        return True
        
    # 3. 拦截极短的无意义碎片（如单纯的 "Fig. 1."）
    if len(text.split()) < 10:
        return True
        
    return False

def count_filtered_stats(base_dir):
    total_papers = 0
    total_paragraphs = 0  # 对应有效 Chunks
    total_sentences = 0
    total_words = 0
    
    print(f"🚀 开始扫描并执行降噪过滤统计...\n")
    
    if not os.path.exists(base_dir):
        print(f"❌ 找不到路径 {base_dir}，请检查配置！")
        return

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_chunks.json"):
                total_papers += 1
                filepath = os.path.join(root, file)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        chunks = json.load(f)
                    except json.JSONDecodeError:
                        continue
                
                # 遍历当前论文的所有 Chunk 进行甄别
                for chunk in chunks:
                    if is_noisy_chunk(chunk):
                        continue # 🌟 命中黑名单，直接跳过，不计入统计！
                    
                    # 统计有效 Paragraphs (Chunks)
                    total_paragraphs += 1
                    
                    # 统计有效 Words
                    words = chunk.split()
                    total_words += len(words)
                    
                    # 统计有效 Sentences (按句号/问号/叹号 + 空白符/行尾 拆分)
                    sentences = re.split(r'[.!?](?:\s+|\n|$)', chunk)
                    valid_sentences = [s for s in sentences if len(s.strip()) > 3]
                    total_sentences += len(valid_sentences)

    # 打印给您写论文用的表格数据
    print("=" * 50)
    print("📊 MCTD-KG 论文数据集统计结果 (已应用黑名单降噪)")
    print("=" * 50)
    print(f"✅ 处理论文总数 (Papers):      {total_papers:,}")
    print(f"✅ 有效段落数 (# Paragraphs):  {total_paragraphs:,} (已过滤版权/参考文献噪音)")
    print(f"✅ 有效句子数 (# Sentences):   {total_sentences:,}")
    print(f"✅ 有效总词数 (# Words):       {total_words:,}")
    print("=" * 50)

if __name__ == "__main__":
    count_filtered_stats(CHUNKS_DIR)