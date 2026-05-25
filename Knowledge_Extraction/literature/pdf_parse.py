import os
import re
import json
import fitz  # PyMuPDF
import traceback

# ================= 配置区 =================
ROOT_DIR = "/data/li-pei-ze/lpz/ra/实验/literature/literature_data"
PDF_DIR = os.path.join(ROOT_DIR, "0_raw_pdfs")
FILTERED_MD_DIR = os.path.join(ROOT_DIR, "2_filtered_markdowns")
CHUNKS_DIR = os.path.join(ROOT_DIR, "3_llm_chunks")
ERROR_LOG_FILE = os.path.join(ROOT_DIR, "failed_fastparse_log.txt")

for d in [FILTERED_MD_DIR, CHUNKS_DIR]:
    os.makedirs(d, exist_ok=True)

# ================= 核心切割逻辑 =================
def fast_extract_and_chunk(pdf_path, chunk_size_words=400):
    doc = fitz.open(pdf_path)
    text_blocks = []
    
    # 1. 粗略读取所有文本块（规避掉极短的杂音，如页码）
    for page in doc:
        blocks = page.get_text("blocks")
        # 按 y 坐标从上到下简单排序
        blocks.sort(key=lambda b: b[1])
        for b in blocks:
            if b[6] == 0:  # 0 表示这是一个纯文本块
                text = b[4].strip()
                # 修复换行符截断单词的问题 (e.g., "pro-\ncessing" -> "processing")
                text = re.sub(r'-\n\s*', '', text)
                text = text.replace('\n', ' ')
                if len(text) > 30: 
                    text_blocks.append(text)
                    
    full_text = "\n\n".join(text_blocks)

    # 2. ✂️ 剁掉尾巴：References & Acknowledgements
    # 寻找文末常见标志的最后一次出现位置（无视严格的换行符）
    tail_keywords = [
        r'\bDeclaration of competing interest\b', 
        r'\bCRediT authorship contribution statement\b',
        r'\bAcknowledgements\b',
        r'\bReferences\b(?=\s*\[1\]|\s*1\.)', # 匹配 References 且后面很快跟着 [1] 或 1.
        r'\n\s*\[1\]\s+[A-Z]\.\s+[A-Za-z]'  # 直接匹配 [1] X. Yyy 这种标准文献开头
    ]
    
    cut_index = len(full_text)
    for pattern in tail_keywords:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        if matches:
            # 找到最靠后的一个匹配项的位置
            match_pos = matches[-1].start()
            if match_pos < cut_index:
                cut_index = match_pos
                
    # 一刀切掉所有标志之后的内容
    if cut_index < len(full_text):
        full_text = full_text[:cut_index]

    # 3. ✂️ 掏空腹部：剪掉 Introduction
    # 寻找 Introduction 的开头
    intro_pattern = r'\n\s*(1\.?\s*)?(Introduction|INTRODUCTION)\s*\n'
    intro_match = re.search(intro_pattern, full_text)
    
    if intro_match:
        # 从 Intro 往后找下一个主要章节（通常是 Experimental, Method, Results）
        next_sec_pattern = r'\n\s*(2\.?\s*)?(Experimental|Method|Results|Materials)\b[^\n]*\n'
        next_sec_match = re.search(next_sec_pattern, full_text[intro_match.end():], re.IGNORECASE)
        
        if next_sec_match:
            # 拼接：Intro 之前的内容(如 Abstract) + 下一个章节之后的内容
            full_text = full_text[:intro_match.start()] + "\n\n" + full_text[intro_match.end() + next_sec_match.start():]

    # 4. 🪓 切分 Chunks
    paragraphs = full_text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_words = 0
    
    for p in paragraphs:
        words = len(p.split())
        if words < 10: continue # 过滤极短的公式残片
        
        if current_words + words < chunk_size_words:
            current_chunk += p + "\n\n"
            current_words += words
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
            current_words = words
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return full_text, chunks

# ================= 主控流水线 =================
def run_fast_pipeline():
    print("🚀 启动极速正则截断流水线 (PyMuPDF版)...\n")
    
    if not os.path.exists(PDF_DIR):
        print(f"❌ 找不到输入目录 {PDF_DIR}！")
        return

    for category in os.listdir(PDF_DIR):
        category_in_path = os.path.join(PDF_DIR, category)
        if not os.path.isdir(category_in_path): continue
            
        print(f"\n📂 处理分类: 【{category}】")
        cat_filtered_dir = os.path.join(FILTERED_MD_DIR, category)
        cat_chunks_dir = os.path.join(CHUNKS_DIR, category)
        os.makedirs(cat_filtered_dir, exist_ok=True)
        os.makedirs(cat_chunks_dir, exist_ok=True)
        
        for filename in os.listdir(category_in_path):
            if not filename.lower().endswith(".pdf"): continue
            paper_id = filename[:-4]
            pdf_path = os.path.join(category_in_path, filename)
            
            chunk_output_file = os.path.join(cat_chunks_dir, f"{paper_id}_chunks.json")
            if os.path.exists(chunk_output_file):
                print(f"  ⏩ [跳过] {paper_id}")
                continue
                
            print(f"  📄 解析: {paper_id} ...", end=" ", flush=True)
            
            try:
                # 秒级解析与切片
                clean_text, chunks = fast_extract_and_chunk(pdf_path)
                
                # 保存去掉头尾的纯文本版本（供核对）
                with open(os.path.join(cat_filtered_dir, f"{paper_id}_clean.txt"), 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                    
                # 保存 JSON Chunks
                with open(chunk_output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                    
                print(f"✅ 成功! -> {len(chunks)} Chunks")
            except Exception as e:
                print(f"❌ 失败!")
                with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"Paper: {paper_id}\nError: {traceback.format_exc()}\n---\n")

if __name__ == "__main__":
    run_fast_pipeline()