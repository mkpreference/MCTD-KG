import json
import numpy as np
import pandas as pd
import faiss
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ================= 基础配置 =================
API_KEY = "sk-fd64f2be977547b4a10ee6a3316021b3"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"
EMBEDDING_MODEL_PATH = os.path.abspath("/root/files_lpz/bge-m3")

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

# 你的文件路径
QA_DATASET_PATH = "/root/files_lpz/ra/实验/literature/qa_dataset/qa_dataset_merged.json"               # 替换为你的测试集路径
DICTIONARY_FILE_PATH = "/root/files_lpz/ra/实验/literature/qa_dataset/材料大辞典语料.xlsx" # 替换为你的大辞典路径
# === [ADD] 文献 chunk 目录 ===
LITERATURE_CHUNKS_DIR = "/root/files_lpz/ra/实验/literature/literature_data/3_llm_chunks"
# ============================
OUTPUT_DIR = "/root/files_lpz/ra/实验/literature/qa_dataset/qa_ans/rag_eval_results"                   # 实验结果保存目录
# ===========================================

class StandardRAGPipeline:
    def __init__(self):
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.embedding_model = SentenceTransformer(
            model_name_or_path=EMBEDDING_MODEL_PATH, 
            device='cuda:1'
        )
        self.index = None
        self.document_chunks = []

    # === [ADD] literature_chunks_dir 参数用于额外加载文献chunk ===
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
        """
        调用 Qwen 大模型评估回答是否正确。
        请将 qwen_client 替换为你实际使用的 LLM 客户端实例。
        """
        # 评估专用提示词
        system_prompt = """你是一个客观、严谨的答案评估专家。
你的任务是根据【问题】和【标准答案】，判断【实际回答】是否正确。
判断标准：
1. 如果【实际回答】包含了【标准答案】的核心语义和事实，即使表述不同，也判定为正确 (YES)。
2. 如果【实际回答】与【标准答案】矛盾、信息缺失、或者提供了完全无关的内容，判定为错误 (NO)。
3. 如果【实际回答】在包含【标准答案】的同时，补充了其他正确相关的上下文信息，判定为正确 (YES)。

请严格按照以下格式输出你的判断结果：
{"judgment": "YES"} 或 {"judgment": "NO"}
不要输出任何其他解释性文字。"""

        user_prompt = f"""
【问题】: {question}
【标准答案】: {gold_answer}
【实际回答】: {llm_answer}
"""

        try:
            # 发起 API 请求
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # 评估任务需要低温度，保持结果稳定
                max_tokens=10    # 评估结果非常简短，不需要太多 token
            )
            
            result_text = response.choices[0].message.content.strip()
            
            import json # 确保顶部引入了 json 模块
            
            # 第一步：字符串清洗。移除模型可能自作主张加上的 Markdown 代码块标记
            clean_text = result_text.replace("```json", "").replace("```", "").strip()
            
            try:
                # 第二步：尝试将其作为标准 JSON 解析
                result_json = json.loads(clean_text)
                return result_json.get("judgment", "NO") == "YES"
                
            except json.JSONDecodeError:
                # 第三步（核心优化）：如果 JSON 解析仍然失败（模型输出格式彻底乱了）
                # 考虑到 max_tokens=10 限制了输出长度，我们可以直接暴力寻找关键字
                text_upper = result_text.upper()
                if "YES" in text_upper:
                    return True
                elif "NO" in text_upper:
                    return False
                else:
                    # 连 YES/NO 都没有，退回到原有的模糊匹配
                    return gold_answer.lower() in llm_answer.lower()
            
        except Exception as e:
            print(f"[评估报错] Qwen 调用失败或解析错误: {e}")
            # 如果大模型调用本身失败（如网络错误、接口超时），为了兜底，退回到模糊匹配
            return gold_answer.lower() in llm_answer.lower()

# ================= 自动化批量测试逻辑 =================
def run_top_k_ablation_experiment(k_values=[10]):  #[1, 3, 5, 10, 15]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(QA_DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    pipeline = StandardRAGPipeline()
    pipeline.build_vector_database(DICTIONARY_FILE_PATH, literature_chunks_dir=LITERATURE_CHUNKS_DIR)  # === [ADD] 传入文献chunk目录 ===
    
    summary_results = {}

    for k in k_values:
        print(f"\n{'='*40}")
        print(f"🚀 开始测试 Standard RAG (Top-K = {k})")
        print(f"{'='*40}")
        
        correct_count = 0
        detailed_logs = []
        
        for item in dataset:
            question = item["question_en"]
            gold_answer = item["gold_answer_en"]
            strategy = item.get("strategy_tested", "未知跳数")
            
            # 1. 检索与生成
            context = pipeline.retrieve_context(question, top_k=k)
            llm_answer = pipeline.generate_answer(question, context)
            
            # 2. 评估
            is_correct = pipeline.judge_with_qwen(question, gold_answer, llm_answer)
            if is_correct:
                correct_count += 1
                
            print(f"[题型: {strategy}] 真实答案: {gold_answer} | 模型输出: {llm_answer} | {'✅' if is_correct else '❌'}")
            
            # 3. 记录日志
            log_item = dict(item)
            log_item.update({
                "top_k_tested": k,
                "retrieved_context": context,
                "llm_prediction": llm_answer,
                "is_correct": is_correct
            })
            detailed_logs.append(log_item)
            
        # 计算该 K 值的准确率
        accuracy = correct_count / len(dataset)
        summary_results[f"Top_{k}"] = accuracy
        print(f"\n>>> Top-{k} 测试完成! 准确率: {accuracy*100:.1f}% ({correct_count}/{len(dataset)})")

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

        # ================= [MODIFIED] 保存该 K 值的详细记录（含统计摘要） =================
        out_file = os.path.join(OUTPUT_DIR, f"rag_result_top_{k}.json")
        output_data = {
            "experiment_summary": {
                "top_k": k,
                "total_questions": len(dataset),
                "correct_count": correct_count,
                "final_accuracy": f"{accuracy*100:.1f}%",
                # === [ADD] 各类型问题正确率 ===
                "accuracy_by_type": {qt: f"{st['correct']/st['total']*100:.2f}%" for qt, st in type_stats.items()},
                # === [ADD] 不同来源正确率 ===
                "accuracy_by_source": {src: f"{ss['correct']/ss['total']*100:.2f}%" for src, ss in source_stats.items()},
            },
            "detailed_results": detailed_logs
        }
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

    # 打印最终对比表格
    print("\n" + "="*40)
    print("🏆 实验总结：不同 Top-K 对比")
    print("="*40)
    for k, acc in summary_results.items():
        print(f"{k.ljust(10)}: Accuracy = {acc*100:.1f}%")

if __name__ == "__main__":
    # 我们测试 1, 3, 5, 10, 15 这几个典型的 K 值
    run_top_k_ablation_experiment(k_values=[5,8,10,15]) #[5, 10, 15]