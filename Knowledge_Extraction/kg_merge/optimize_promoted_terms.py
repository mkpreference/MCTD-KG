import json
from neo4j import GraphDatabase
from openai import OpenAI
import datetime
import re

class PromotedTermOptimizer:
    def __init__(self, uri, user, password, api_key, base_url):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def close(self):
        self.driver.close()

    def _log_optimization(self, message, log_file="/root/files_lpz/ra/实验/literature/extract/merge/log_addliterature/term_optimization_log.txt"):
        """将名称优化过程的双向日志记录到本地文件"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_message + "\n")

    def _ask_llm_for_name_optimization(self, name, context_info):
        """调用 LLM 判断并优化术语名称"""
        prompt = f"""
你是一位严谨的材料科学术语词典编纂专家。
以下是一个刚刚从学术文献提取并晋升为候选专业术语的名称。由于是机器抽取，它可能带有文献叙述时的冗余词汇、动词前缀、单复数不当或格式不规范。

【当前节点名称】: "{name}"
【在图谱中的上下文关联】: {context_info}

【审查与优化准则】：
1. 剥离冗余修饰：如果名称包含动作、状态或应用描述（如 "deposition of hydroxyapatite" -> "hydroxyapatite", "using steel" -> "steel", "titanium alloy parts" -> "titanium alloy"），请将其还原为最核心的材料学本源实体。
2. 规范单复数：材料术语一般使用单数（除非特殊惯用复数）。
3. 纠正拼写/大小写：统一为最标准的学术拼写。
4. 保持原样：如果该名称本身已经是一个非常标准、干净的材料学专业术语（如 "titanium alloy", "Al2O3", "Fourier transform infrared spectroscopy"），请保持原样。

任务：请判断该名称是否符合严格的规范。如果符合，保持原样；如果不符合，给出优化后的干净术语名称。

严格按照以下 JSON 格式输出：
{{
  "is_standard": true/false,
  "optimized_name": "优化后的名称（如果 is_standard 为 true，请填入原名称）",
  "reason": "简要的判断或修改理由（限30字）"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "system", "content": "你是一个严谨的材料学术语规范助手，只输出标准 JSON。"},
                          {"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw_content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"is_standard": True, "optimized_name": name, "reason": "解析失败，默认保留原样"}
        except Exception as e:
            self._log_optimization(f"   ❌ [LLM 请求异常]: {e}")
            return {"is_standard": True, "optimized_name": name, "reason": "调用异常，默认保留原样"}

    def run_optimization(self):
        """执行全量的术语名称优化流程"""
        self._log_optimization("\n==================================================")
        self._log_optimization("🚀 开始执行：晋升术语名称智能规范化审查")
        self._log_optimization("==================================================")

        # 1. 查询所有已晋升但尚未被审查过名称的 Term 节点
        fetch_query = """
        MATCH (n:Term)
        WHERE n.is_promoted = true AND n.name_optimized IS NULL
        
        OPTIONAL MATCH (n)-[r]-(neighbor)
        WITH n, collect(type(r) + ' ' + neighbor.name)[0..5] AS context_info
        
        RETURN id(n) AS node_id, n.name AS name, context_info
        """

        optimized_count = 0
        kept_count = 0

        with self.driver.session() as session:
            candidates = session.run(fetch_query).data()
            
            if not candidates:
                self._log_optimization("✅ 当前图谱中没有需要优化的待审查晋升节点。")
                return

            self._log_optimization(f"🔍 发现 {len(candidates)} 个待审查的晋升术语节点...\n")

            for cand in candidates:
                node_id = cand["node_id"]
                original_name = cand["name"]
                context_info = cand["context_info"]

                self._log_optimization(f"▶ 正在审查: [{original_name}]")
                decision = self._ask_llm_for_name_optimization(original_name, context_info)

                is_standard = decision.get("is_standard", True)
                optimized_name = decision.get("optimized_name", original_name)
                reason = decision.get("reason", "")

                if not is_standard and optimized_name != original_name:
                    self._log_optimization(f"   🛠️ [执行优化]: '{original_name}'  --->  '{optimized_name}'")
                    self._log_optimization(f"   💡 [LLM 理由]: {reason}")
                    
                    # 更新节点名称，并保留原名作为溯源属性
                    update_query = """
                    MATCH (n:Term) WHERE id(n) = $node_id
                    SET n.name = $optimized_name,
                        n.original_name_before_opt = $original_name,
                        n.name_optimized = true,
                        n.name_opt_date = datetime()
                    """
                    session.run(update_query, node_id=node_id, optimized_name=optimized_name, original_name=original_name)
                    optimized_count += 1
                else:
                    self._log_optimization(f"   ✅ [符合规范]: 保持原样。({reason})")
                    # 打上已审查标记，防止下次重复审查
                    mark_query = """
                    MATCH (n:Term) WHERE id(n) = $node_id
                    SET n.name_optimized = true
                    """
                    session.run(mark_query, node_id=node_id)
                    kept_count += 1

                self._log_optimization("-" * 40)

        # 打印最终统计
        self._log_optimization("\n📊 名称优化统计结果：")
        self._log_optimization(f"  - 发现不规范并成功优化改名的节点：{optimized_count} 个")
        self._log_optimization(f"  - 原本就符合规范保持原样的节点：{kept_count} 个")
        self._log_optimization("================ 优化流程结束 ================\n")

# ================= 运行入口 =================
if __name__ == "__main__":
    optimizer = PromotedTermOptimizer(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="LPZ12256zpl",
        api_key="sk-fd64f2be977547b4a10ee6a3316021b3",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    optimizer.run_optimization()
    optimizer.close()