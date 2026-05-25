import os
import re
import json
import subprocess
import traceback
import shutil  # 必须导入这个自带的路径寻址库

# ================= 配置区 =================
ROOT_DIR = "/data/li-pei-ze/lpz/ra/实验/literature/literature_data"

# 输入与输出目录
PDF_DIR = os.path.join(ROOT_DIR, "0_raw_pdfs")
RAW_MD_DIR = os.path.join(ROOT_DIR, "1_raw_markdowns")
FILTERED_MD_DIR = os.path.join(ROOT_DIR, "2_filtered_markdowns")
CHUNKS_DIR = os.path.join(ROOT_DIR, "3_llm_chunks")

# 错误日志
ERROR_LOG_FILE = os.path.join(ROOT_DIR, "failed_preprocessing_log.txt")

# 确保输出目录存在
for d in [RAW_MD_DIR, FILTERED_MD_DIR, CHUNKS_DIR]:
    os.makedirs(d, exist_ok=True)


# ================= 模块 1：PDF 转 Markdown (Marker) =================
def convert_pdf_to_md(pdf_path, output_dir):
    """
    调用开源的 Marker 工具，将单篇 PDF 转换为 Markdown
    注意：Marker 会在 output_dir 下自动创建一个与 pdf 同名的文件夹，里面包含 .md 和图片
    """
    try:
        # 使用 marker_single 命令行工具
        cmd = f'marker_single "{pdf_path}" --out_folder "{output_dir}" --batch_procs 2'
        
        # 运行子进程，隐藏多余的输出以保持控制台干净
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Marker 转换失败: {result.stderr}")
            
        return True
    except Exception as e:
        raise RuntimeError(f"PDF转换异常: {str(e)}")


def batch_convert_category_pdfs(input_dir, output_dir):
    """
    火力全开模式：使用 marker 原生批处理命令处理整个文件夹
    【终极参数修正版：增加 --output_dir 标签】
    """
    try:
        workers = 6  # 并发数
        
        marker_exe = shutil.which("marker")
        if marker_exe is None:
            raise RuntimeError("当前环境中找不到 marker 命令！")
            
        print(f"    ⚡ 正在启动 GPU 并发加速转换 (Workers: {workers})...")
        print(f"    🔧 自动寻址 Marker 绝对路径: {marker_exe}") 
        
        # ================= 核心修复 =================
        # 给输出文件夹加上明确的参数标签 --output_dir 
        # (如果后续提示找不到该参数，可以尝试换成 --out_folder)
        cmd = f'"{marker_exe}" "{input_dir}" --output_dir "{output_dir}" --workers {workers}'
        
        # 运行子进程
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode != 0:
            raise RuntimeError("Marker 批量转换发生错误！请查看上方控制台报错。")
            
    except Exception as e:
        raise RuntimeError(f"批量转换异常: {str(e)}")

# ================= 模块 2：清洗与语义切片 =================
def process_markdown(md_content, chunk_size_words=400):
    """提取高价值章节，并返回纯净版文本和 LLM Chunks"""
    # 匹配 1-3 级标题，如 # 标题, ## 标题, ### 标题
    sections = re.split(r'(^#{1,3}\s+.*)', md_content, flags=re.MULTILINE)
    
    # 关键词策略（全小写比对）
    keep_keywords = ['abstract', 'method', 'experimental', 'result', 'discussion', 'conclusion']
    drop_keywords = ['introduction', 'background', 'reference', 'acknowledgement']

    filtered_text = ""
    chunks = []
    
    current_chunk_text = ""
    current_chunk_word_count = 0

    # sections 分割后，奇数索引是标题，偶数索引是内容
    for i in range(1, len(sections), 2):
        header = sections[i].lower()
        body = sections[i+1].strip() if i+1 < len(sections) else ""
        
        # 判断：如果标题包含保留词，且不包含丢弃词
        if any(k in header for k in keep_keywords) and not any(d in header for d in drop_keywords):
            filtered_text += sections[i] + "\n" + body + "\n\n"
            
            # 按段落 (\n\n) 进行细粒度切分
            paragraphs = body.split('\n\n')
            for p in paragraphs:
                words = len(p.split())
                if words < 10: 
                    continue  # 过滤极短的公式残片或图表说明噪音
                
                # 累加单词直到达到 chunk 大小
                if current_chunk_word_count + words < chunk_size_words:
                    current_chunk_text += p + "\n\n"
                    current_chunk_word_count += words
                else:
                    chunks.append(current_chunk_text.strip())
                    current_chunk_text = p + "\n\n"
                    current_chunk_word_count = words
                    
    # 把最后一个没装满的 chunk 也加进去
    if current_chunk_text:
        chunks.append(current_chunk_text.strip())
        
    return filtered_text, chunks

def run_fast_preprocessing_pipeline():
    print("🚀 开始 GPU 火力全开版流水线 (按文件夹批处理)...\n")
    
    if not os.path.exists(PDF_DIR):
        print(f"❌ 找不到输入目录 {PDF_DIR}！")
        return

    for category in os.listdir(PDF_DIR):
        category_in_path = os.path.join(PDF_DIR, category)
        if not os.path.isdir(category_in_path): 
            continue
            
        print(f"\n========================================")
        print(f"📂 开始处理分类: 【{category}】")
        print(f"========================================")
        
        cat_md_dir = os.path.join(RAW_MD_DIR, category)
        cat_filtered_dir = os.path.join(FILTERED_MD_DIR, category)
        cat_chunks_dir = os.path.join(CHUNKS_DIR, category)
        
        os.makedirs(cat_md_dir, exist_ok=True)
        os.makedirs(cat_filtered_dir, exist_ok=True)
        os.makedirs(cat_chunks_dir, exist_ok=True)

        # ================= STEP A: 文件夹级 PDF 转 MD =================
        # 直接把整个分类文件夹丢进去，1 次跑完该分类下所有 PDF！
        try:
            batch_convert_category_pdfs(category_in_path, cat_md_dir)
        except Exception as e:
            print(f"❌ 分类 {category} 的 PDF 转换失败: {e}")
            continue # 如果转换失败，跳过后续切片，继续下一个分类

        # ================= STEP B: 遍历生成的 MD 进行切片 =================
        # Marker 处理完后，会在 cat_md_dir 下为每篇论文生成一个同名文件夹
        print(f"\n    ✂️ 正在进行语义切片...")
        for paper_folder in os.listdir(cat_md_dir):
            paper_md_dir = os.path.join(cat_md_dir, paper_folder)
            if not os.path.isdir(paper_md_dir):
                continue
                
            paper_id = paper_folder
            md_file_path = os.path.join(paper_md_dir, f"{paper_id}.md")
            chunk_output_file = os.path.join(cat_chunks_dir, f"{paper_id}_chunks.json")
            
            # 断点续传：已切片则跳过
            if os.path.exists(chunk_output_file):
                continue 

            if not os.path.exists(md_file_path):
                print(f"  ⚠️ 找不到 {paper_id} 的 MD 文件，可能转换失败。")
                continue
                
            # 读取、过滤、切片、保存 (这里调用您之前的 process_markdown 函数)
            try:
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    raw_md = f.read()
                
                filtered_md, chunks = process_markdown(raw_md) # 依赖之前的代码
                
                # 保存纯净版
                with open(os.path.join(cat_filtered_dir, f"{paper_id}_filtered.md"), 'w', encoding='utf-8') as f:
                    f.write(filtered_md)
                    
                # 保存 Chunks
                with open(chunk_output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                    
                print(f"  ✅ {paper_id} 切片完成 -> 产出 {len(chunks)} 个数据块")
            except Exception as e:
                print(f"  ❌ {paper_id} 切片报错: {e}")

if __name__ == "__main__":
    run_fast_preprocessing_pipeline()

# ================= 主控流水线 =================
def run_preprocessing_pipeline():
    print("🚀 开始构建 MCTD-KG 预处理流水线 (PDF -> MD -> Chunks)...\n")
    
    if not os.path.exists(PDF_DIR):
        print(f"❌ 找不到输入目录 {PDF_DIR}，请确保您已准备好 PDF 文件夹！")
        return

    # 1. 遍历 11 个大类文件夹
    for category in os.listdir(PDF_DIR):
        category_in_path = os.path.join(PDF_DIR, category)
        if not os.path.isdir(category_in_path): 
            continue
            
        print(f"\n📂 开始处理分类: 【{category}】")
        
        # 为当前分类创建输出子目录
        cat_md_dir = os.path.join(RAW_MD_DIR, category)
        cat_filtered_dir = os.path.join(FILTERED_MD_DIR, category)
        cat_chunks_dir = os.path.join(CHUNKS_DIR, category)
        
        os.makedirs(cat_md_dir, exist_ok=True)
        os.makedirs(cat_filtered_dir, exist_ok=True)
        os.makedirs(cat_chunks_dir, exist_ok=True)
        
        # 2. 遍历该分类下的所有 PDF 论文
        for filename in os.listdir(category_in_path):
            if not filename.lower().endswith(".pdf"): 
                continue
                
            paper_id = filename[:-4] # 去掉 .pdf 后缀
            pdf_path = os.path.join(category_in_path, filename)
            
            # 防御性编程：检查是否已经处理过（断点续传）
            chunk_output_file = os.path.join(cat_chunks_dir, f"{paper_id}_chunks.json")
            if os.path.exists(chunk_output_file):
                print(f"  ⏩ [跳过] {paper_id} 的 Chunks 已存在。")
                continue
                
            print(f"  📄 正在处理: {paper_id} ...", end=" ", flush=True)
            
            try:
                # ================= STEP A: PDF 转 MD =================
                # 注意：Marker 会在 cat_md_dir 下生成一个名为 paper_id 的文件夹
                convert_pdf_to_md(pdf_path, cat_md_dir)
                
                # 寻找 Marker 生成的 md 文件路径
                marker_output_dir = os.path.join(cat_md_dir, paper_id)
                md_file_path = os.path.join(marker_output_dir, f"{paper_id}.md")
                
                if not os.path.exists(md_file_path):
                    raise FileNotFoundError("Marker 未能生成预期的 .md 文件。")
                    
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    raw_md = f.read()

                # ================= STEP B: 清洗与切片 =================
                filtered_md, chunks = process_markdown(raw_md)
                
                if len(chunks) == 0:
                    raise ValueError("未能从文章中提取到任何有效切片 (可能章节未正确解析)。")

                # ================= STEP C: 分别保存中间态 =================
                # 1. 保存纯净版 Markdown (方便人工查阅)
                with open(os.path.join(cat_filtered_dir, f"{paper_id}_filtered.md"), 'w', encoding='utf-8') as f:
                    f.write(filtered_md)
                    
                # 2. 保存供大模型抽取的 Chunk 列表 (JSON 格式)
                with open(chunk_output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                    
                print(f"✅ 成功! 产出 {len(chunks)} 个数据块。")
                
            except Exception as e:
                # 捕获异常，记录死信队列，继续下一篇
                print(f"❌ 失败!")
                with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"Category: {category} | Paper: {paper_id}\n")
                    f.write(f"Error: {traceback.format_exc()}\n")
                    f.write("-" * 50 + "\n")

    print("\n🎉 全部预处理完成！请前往 2_filtered_markdowns 和 3_llm_chunks 检查数据质量。")
    print(f"（如有解析失败的论文，请查看 {ERROR_LOG_FILE}）")

if __name__ == "__main__":
    # run_preprocessing_pipeline()
    run_fast_preprocessing_pipeline()