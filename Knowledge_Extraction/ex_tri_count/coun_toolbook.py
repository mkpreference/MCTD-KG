import pandas as pd
import re

# ================= 配置区 =================
# 将这里替换为您大辞典文件的实际路径 (支持 .xlsx 或 .csv)
DICT_FILE_PATH = "/data/li-pei-ze/lpz/ra/实验/数据集构建/数据集术语选取/材料大辞典语料.xlsx" 

def count_dictionary_stats(file_path):
    print(f"🚀 开始读取大辞典数据: {file_path} ...\n")
    
    # 兼容 CSV 或 Excel
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
        
    total_paragraphs = 0  # 对应有效词条数
    total_sentences = 0
    total_words = 0
    
    # 遍历表格的每一行
    for index, row in df.iterrows():
        # 获取英文名称和英文释义（转为字符串并去除首尾空格）
        name_en = str(row.get('name_en', '')).strip()
        explanation_en = str(row.get('explanation_en', '')).strip()
        
        # 数据清洗：如果某一行为空、nan，或者没有英文释义，则跳过不计
        if name_en.lower() == 'nan' or explanation_en.lower() == 'nan' or not name_en or not explanation_en:
            continue
            
        # 按照您喂给 LLM 的格式进行拼接
        text_block = f"{name_en}: {explanation_en}"
        
        # 1. 统计 Paragraphs (每一个有效的术语条目记为 1 个 Paragraph/Chunk)
        total_paragraphs += 1
        
        # 2. 统计 Words (按空格切分单词)
        words = text_block.split()
        total_words += len(words)
        
        # 3. 统计 Sentences (按句号、问号、叹号切分)
        # 注意：英文中常有简写如 "e.g.", "etc." 可能会造成轻微过度切分，这里采用标准的粗略切分规则
        sentences = re.split(r'[.!?](?:\s+|\n|$)', text_block)
        valid_sentences = [s for s in sentences if len(s.strip()) > 2] # 过滤掉切出的空字符或极短标点
        
        # 如果释义只是一个没有句号的短语（比如 "See Armco-iron" 如果没带句号），也要算作 1 个语义句子
        num_sentences = len(valid_sentences)
        if num_sentences == 0 and len(text_block.strip()) > 0:
            num_sentences = 1
            
        total_sentences += num_sentences

    print("=" * 50)
    print("📊 MCTD-KG 大辞典数据源 (Tool book) 统计结果")
    print("=" * 50)
    print(f"✅ 有效词条数 (# Paragraphs):  {total_paragraphs:,}")
    print(f"✅ 有效句子数 (# Sentences):   {total_sentences:,}")
    print(f"✅ 有效总词数 (# Words):       {total_words:,}")
    print("=" * 50)
    
    # 返回结果供后续使用
    return total_paragraphs, total_sentences, total_words

if __name__ == "__main__":
    count_dictionary_stats(DICT_FILE_PATH)