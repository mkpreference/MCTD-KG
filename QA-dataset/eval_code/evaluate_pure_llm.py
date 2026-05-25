import json
import os
from openai import OpenAI

# ================= 基础配置 (严格保持与前面实验一致) =================
API_KEY = "sk-24e79ff04c90448a9e4ad2338f56b8e7"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"

QA_DATASET_PATH = "/root/files_lpz/ra/实验/literature/qa_dataset/qa_dataset_merged.json"               # 替换为你的测试集路径
OUTPUT_FILE = "/root/files_lpz/ra/实验/literature/qa_dataset/qa_ans/pure_llm_eval_results.json"        # 实验结果保存路径
# =========================================================================

class PureLLMPipeline:
    def __init__(self):
        print(f"[基线系统日志] 正在初始化 Pure LLM ({MODEL_NAME}) 客户端...")
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def generate_answer(self, question):
        """
        纯大模型生成：没有任何外部知识上下文，直接“闭卷考试”。
        """
        prompt = f"""
你是一个专业的材料科学专家。请依靠你内部的知识回答以下材料科学问题。

【约束条件】：
1. 你的回答必须极其精简，只能输出特定的术语名称或分类名称。
2. 绝对不要输出任何推理过程、解释性文字或标点符号。

[用户问题]:
{question}
"""
        try:
            completion = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个严谨的材料科学助手，严格遵守输出格式限制。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.01  # 保持极低温度，测试其固化的内部知识
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[生成报错] {e}")
            return "Generation_Error"

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
def run_pure_llm_experiment():
    print("\n" + "="*50)
    print("🚀 开始执行 Pure LLM (闭卷) 基线评测")
    print("="*50)
    
    with open(QA_DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    pipeline = PureLLMPipeline()
    
    correct_count = 0
    detailed_logs = []
    
    for item in dataset:
        question = item["question_en"]
        gold_answer = item["gold_answer_en"]
        strategy = item.get("strategy_tested", "未知跳数")
        
        # 1. 闭卷生成
        llm_answer = pipeline.generate_answer(question)
        
        # 2. 评估结果
        is_correct = pipeline.judge_with_qwen(question, gold_answer, llm_answer)
        if is_correct:
            correct_count += 1
            
        print(f"[题型: {strategy}]")
        print(f"  问题: {question}")
        print(f"  真实答案: {gold_answer} | 模型输出: {llm_answer}")
        print(f"  状态: {'[通过 ✅]' if is_correct else '[失败 ❌]'}\n")
        
        # 3. 记录日志
        log_item = dict(item)
        log_item.update({
            "method": "Pure_LLM",
            "llm_prediction": llm_answer,
            "is_correct": is_correct
        })
        detailed_logs.append(log_item)
        
    # 计算准确率
    accuracy = correct_count / len(dataset)
    print("="*50)
    print(f"🏆 Pure LLM 评测结束! 最终准确率: {accuracy*100:.1f}% ({correct_count}/{len(dataset)})")

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

    print("="*50)

    # ================= [MODIFIED] 保存记录（含统计摘要） =================
    output_data = {
        "experiment_summary": {
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
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"详细的实验结果记录已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pure_llm_experiment()