import json
import numpy as np
import pandas as pd
import faiss
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ================= 1. 基础与路径配置 =================
API_KEY = "sk-fd64f2be977547b4a10ee6a3316021b3"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"
EMBEDDING_MODEL_PATH = os.path.abspath("/root/files_lpz/bge-m3")

# 【请确认以下路径是否为你最新构建好的数据集路径】
# 你的文件路径
QA_DATASET_PATH = "/root/files_lpz/ra/实验/literature/qa_dataset/qa_dataset_merged.json"               # 替换为你的测试集路径
DICTIONARY_FILE_PATH = "/root/files_lpz/ra/实验/literature/qa_dataset/材料大辞典语料.xlsx" # 替换为你的大辞典路径
# === [ADD] 文献 chunk 目录 ===
LITERATURE_CHUNKS_DIR = "/root/files_lpz/ra/实验/literature/literature_data/3_llm_chunks"
# ============================
OUTPUT_DIR = "/root/files_lpz/ra/实验/literature/qa_dataset/qa_ans/rag_eval_results"                   # 实验结果保存目录                

# ================= 2. 核心 RAG 管道 =================

# === [ADD] 文献 chunk 降噪黑名单 ===
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

def _is_noisy_chunk(chunk):
    """过滤非材料知识的噪音 chunk（版权声明、参考文献、极短碎片）"""
    import re
    text = chunk.strip()
    if any(bad_word in text for bad_word in BLACKLIST):
        return True
    if re.match(r'^\[\d+\]', text) or len(re.findall(r'\[\d+\]', text)) > 6:
        return True
    if len(text.split()) < 10:
        return True
    return False
# ====================================
class StandardRAGPipeline:
    def __init__(self):
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=EMBEDDING_MODEL_PATH, 
            device='cuda:1'
        )
        self.index = None
        self.document_chunks = []

    def build_vector_database(self, xlsx_path, cache_dir="/root/files_lpz/ra/实验/literature/exp_code/rag_cache", force_rebuild=False, literature_chunks_dir=None):
        """
        建立基于实体级分块的 Faiss 向量数据库，支持本地缓存和强制重建。
        :param literature_chunks_dir: 可选的文献 JSON chunk 目录（递归扫描所有 *_chunks.json）
        """
        import pickle
        import glob
        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, "faiss.index")
        chunks_path = os.path.join(cache_dir, "chunks.pkl")

        # 【核心缓存逻辑】：如果不强制重建，且缓存文件存在，则直接秒级加载
        if not force_rebuild and os.path.exists(index_path) and os.path.exists(chunks_path):
            print(f"[基线系统日志] ⚡ 检测到本地缓存，直接加载向量库和文本块...")
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                self.document_chunks = pickle.load(f)
            print(f"[基线系统日志] 加载成功！共包含 {len(self.document_chunks)} 个片段。")
            return

        # ================= 如果没有缓存或要求强制重建，则执行完整的构建流程 =================
        print(f"[基线系统日志] 正在读取并分块 {xlsx_path} ...")
        df = pd.read_excel(xlsx_path).dropna(subset=['name_en', 'explanation_en'])

        self.document_chunks = [] # 清空旧数据
        for _, row in df.iterrows():
            chunk_text = f"Term: {str(row['name_en']).strip()}\nExplanation: {str(row['explanation_en']).strip()}"
            self.document_chunks.append(chunk_text)

        # ================= [ADD] 加载文献 chunk (JSON 数组) =================
        if literature_chunks_dir and os.path.isdir(literature_chunks_dir):
            json_files = glob.glob(os.path.join(literature_chunks_dir, "**", "*_chunks.json"), recursive=True)
            print(f"[基线系统日志] 正在加载文献 chunk 文件 (共 {len(json_files)} 个)...")
            lit_count = 0
            for jf in json_files:
                with open(jf, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                for chunk_text in chunks:
                    # === [ADD] 过滤噪音 chunk ===
                    if _is_noisy_chunk(chunk_text):
                        continue
                    self.document_chunks.append(f"Literature: {chunk_text.strip()}")
                    lit_count += 1
            print(f"[基线系统日志] 文献 chunk 加载完成，新增 {lit_count} 条，总计 {len(self.document_chunks)} 条。")
        # =================================================================

        print(f"[基线系统日志] 正在为 {len(self.document_chunks)} 个文本块计算向量 (可能需要一些时间)...")
        import torch; torch.cuda.empty_cache()  # === [ADD] 清理缓存碎片，降低 OOM 风险 ===
        embeddings = self.embedding_model.encode(self.document_chunks, batch_size=64, show_progress_bar=True)  # === [OPT] 调大 batch_size 加速（128 超显存故改为 64） ===
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # 将构建好的索引和文本保存到本地
        print("[基线系统日志] 正在将向量库保存至本地缓存...")
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.document_chunks, f)
        print("[基线系统日志] 向量库构建并缓存完成！")

    def retrieve_context(self, question, top_k):
        """执行相似度检索"""
        question_embedding = self.embedding_model.encode([question]).astype('float32')
        distances, indices = self.index.search(question_embedding, top_k)
        
        retrieved_texts = []
        for idx in indices[0]:
            if idx != -1:
                retrieved_texts.append(self.document_chunks[idx])
                
        return "\n\n".join([f"[检索片段 {i+1}]:\n{txt}" for i, txt in enumerate(retrieved_texts)])

    def generate_answer(self, question, context):
        """基于检索到的片段生成答案"""
        prompt = f"""
你是一个严谨的材料科学专家。请仅根据提供的[检索文档片段]回答问题。
【约束】：
1. 严禁使用外部先验知识，必须从片段中找答案。
2. 答案必须极其精简，只输出术语名称或分类名称。
[检索片段]:
{context}

[用户问题]:
{question}
"""
        completion = self.llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个基于检索事实的助手，不编造信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01
        )
        return completion.choices[0].message.content.strip()

    def judge_with_qwen(self, question, gold_answer, llm_answer, model_name="qwen-plus"):
        """调用大模型评估回答是否正确"""
        system_prompt = """你是一个客观、严谨的答案评估专家。
你的任务是根据【问题】和【标准答案】，判断【实际回答】是否正确。
判断标准：
1. 如果【实际回答】包含了【标准答案】的核心语义和事实，即使表述不同，也判定为正确 (YES)。
2. 如果【实际回答】与【标准答案】矛盾、信息缺失、或者提供了完全无关的内容，判定为错误 (NO)。
3. 如果【实际回答】在包含【标准答案】的同时，补充了其他正确相关的上下文信息，判定为正确 (YES)。

请严格按照以下格式输出你的判断结果：
{"judgment": "YES"} 或 {"judgment": "NO"}
不要输出任何其他解释性文字。"""

        user_prompt = f"【问题】: {question}\n【标准答案】: {gold_answer}\n【实际回答】: {llm_answer}"

        try:
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, 
                max_tokens=10    
            )
            
            result_text = response.choices[0].message.content.strip()
            clean_text = result_text.replace("```json", "").replace("```", "").strip()
            
            try:
                result_json = json.loads(clean_text)
                return result_json.get("judgment", "NO") == "YES"
            except json.JSONDecodeError:
                text_upper = result_text.upper()
                if "YES" in text_upper: return True
                elif "NO" in text_upper: return False
                else: return gold_answer.lower() in llm_answer.lower()
        except Exception as e:
            print(f"[评估报错]: {e}")
            return gold_answer.lower() in llm_answer.lower()

# ================= 3. 运行 Top-10 评估并定期保存 =================
def run_rag_evaluation(top_k=10):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"rag_evaluation_results_top{top_k}.json")
    
    print(f"读取测试集: {QA_DATASET_PATH} ...")
    with open(QA_DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    total_questions = len(dataset)

    pipeline = StandardRAGPipeline()
    pipeline.build_vector_database(DICTIONARY_FILE_PATH, literature_chunks_dir=LITERATURE_CHUNKS_DIR)  # === [ADD] 传入文献chunk目录 ===
   
    
    print(f"\n{'='*50}")
    print(f"🚀 开始执行标准 RAG 评测 (检索 Top-K = {top_k})")
    print(f"🎯 共计 {total_questions} 个问题，设置每 50 题自动保存一次。")
    print(f"{'='*50}")
    
    correct_count = 0
    detailed_logs = []
    
    # 引入 enumerate 来追踪当前到第几题了
    for idx, item in enumerate(dataset):
        question = item.get("question_en", item.get("question", ""))
        gold_answer = item.get("gold_answer_en", item.get("gold_answer", ""))
        strategy = item.get("strategy_tested", item.get("category", "未知类型"))
        
        # 1. 检索与生成
        context = pipeline.retrieve_context(question, top_k=top_k)
        llm_answer = pipeline.generate_answer(question, context)
        
        # 2. 评估
        is_correct = pipeline.judge_with_qwen(question, gold_answer, llm_answer)
        if is_correct:
            correct_count += 1
            
        print(f"[{idx + 1}/{total_questions}] [题型: {strategy}] 预期: {gold_answer} | 输出: {llm_answer} | {'✅' if is_correct else '❌'}")
        
        # 3. 组装数据
        log_item = {
            "question": question,
            "gold_answer": gold_answer,
            "question_type": strategy,
            "top_k_used": top_k,
            "retrieved_context": context,
            "llm_prediction": llm_answer,
            "is_correct": is_correct
        }
        for k, v in item.items():
            if k not in log_item:
                log_item[k] = v
                
        detailed_logs.append(log_item)
        
        # ================= 关键新增：每 50 题自动保存一次 =================
        if (idx + 1) % 50 == 0:
            current_acc = correct_count / (idx + 1)
            print(f"\n💾 [Checkpoint] 已处理 {idx + 1} 题，当前准确率: {current_acc*100:.2f}%。正在将进度写入本地文件...")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "experiment_summary": {
                        "top_k": top_k,
                        "total_questions": total_questions,
                        "processed_so_far": idx + 1,
                        "correct_count": correct_count,
                        "current_accuracy": f"{current_acc*100:.2f}%"
                    },
                    "detailed_results": detailed_logs
                }, f, ensure_ascii=False, indent=4)
            print("✅ 进度保存成功！继续执行...\n")
        # =================================================================
        
    # 循环完全结束后，做最后一次最终保存
    accuracy = correct_count / total_questions
    print(f"\n🎉 评测全部完成! 最终准确率: {accuracy*100:.2f}% ({correct_count}/{total_questions})")

    # ================= [ADD] 按类型和来源统计正确率 =================
    type_stats = {}
    source_stats = {}
    for r in detailed_logs:
        qtype = r.get("strategy_tested", "unknown")
        if qtype not in type_stats:
            type_stats[qtype] = {"correct": 0, "total": 0}
        type_stats[qtype]["total"] += 1
        if r.get("is_correct"):
            type_stats[qtype]["correct"] += 1

        src = r.get("source", "unknown")
        if src not in source_stats:
            source_stats[src] = {"correct": 0, "total": 0}
        source_stats[src]["total"] += 1
        if r.get("is_correct"):
            source_stats[src]["correct"] += 1

    print("\n========== 各类型问题正确率 ==========")
    for qtype in ["属性", "0跳", "1跳", "2跳", "3跳", "4跳"]:
        st = type_stats.get(qtype)
        if st:
            acc = st["correct"] / st["total"] * 100
            print(f"  {qtype}: {acc:.2f}% ({st['correct']}/{st['total']})")
        else:
            print(f"  {qtype}: 无数据")

    print("\n========== 不同来源正确率 ==========")
    for src in ["literature", "toolbook"]:
        ss = source_stats.get(src)
        if ss:
            acc = ss["correct"] / ss["total"] * 100
            print(f"  {src}: {acc:.2f}% ({ss['correct']}/{ss['total']})")
        else:
            print(f"  {src}: 无数据")
    # ===============================================================

    print(f"正在覆盖保存完整结果至: {out_file} ...")

    # ================= 🌟 修改：输出包含最终准确率的结构 =================
    final_output = {
        "experiment_summary": {
            "top_k": top_k,
            "total_questions": total_questions,
            "correct_count": correct_count,
            "final_accuracy": f"{accuracy*100:.2f}%",
            # === [ADD] 各类型问题正确率 ===
            "accuracy_by_type": {qt: f"{st['correct']/st['total']*100:.2f}%" for qt, st in type_stats.items()},
            # === [ADD] 不同来源正确率 ===
            "accuracy_by_source": {src: f"{ss['correct']/ss['total']*100:.2f}%" for src, ss in source_stats.items()},
        },
        "detailed_results": detailed_logs
    }
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    # ====================================================================
        
    print("🏆 完整结果落盘完毕！")

if __name__ == "__main__":
    run_rag_evaluation(top_k=10)