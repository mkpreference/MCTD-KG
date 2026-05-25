import json
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz  # 推荐使用 rapidfuzz，比 fuzzywuzzy 更快
from openai import OpenAI
import datetime
import os
import re
class EntityAligner:
    def __init__(self, uri, user, password, api_key, base_url):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def close(self):
        self.driver.close()

    def _log_action(self, msg, log_file="/root/files_lpz/ra/实验/literature/extract/merge/log_lk/entity_linking_log1.txt"):
        """同时将信息输出到控制台（终端）并追加到本地日志文件中"""
        print(msg)  # 终端打印
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {msg}\n")  # 日志文件记录

    def _log_merge(self, original_names, final_name, reasoning="", log_file="/root/files_lpz/ra/实验/literature/extract/merge/log_lk/entity_linking_log2.txt"):
        """将节点合并的详细记录追加到本地日志文件中，包含 LLM 推理过程"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 将理由整合进这条合并记录中
        if reasoning:
            log_msg = f"[{timestamp}] 🔗 合并动作 | 参与合并的原始节点: {original_names}  ===>  最终生成的节点: [{final_name}] | 💡 LLM理由: {reasoning}\n"
        else:
            log_msg = f"[{timestamp}] 🔗 合并动作 | 参与合并的原始节点: {original_names}  ===>  最终生成的节点: [{final_name}]\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg)

    def get_all_entities(self):
        """获取所有待对齐的 Term 和 Other 节点及其上下文"""
        query = """
        MATCH (n)
        WHERE n:Term OR n:Other
        OPTIONAL MATCH (n)-[r]-(neighbor)
        RETURN n.name AS name, 
               labels(n) AS labels, 
               n.source_category AS category, 
               collect(DISTINCT neighbor.name)[0..5] AS neighbors,
               id(n) AS node_id
        """
        with self.driver.session() as session:
            return session.run(query).data()

    def _ask_llm_for_canonical_form(self, cluster_data):
        """调用 LLM 判断聚类中的实体是否可以合并，【引入材料学深度语义规则与 CoT 思维链】"""
        cluster_info = ""
        for i, item in enumerate(cluster_data):
            cluster_info += f"实体{i}: [{item['name']}] (分类: {item['category']})\n"

        prompt = f"""
你是一个严谨的材料科学知识图谱构建专家。
以下是一组基于相似度聚类出来的实体列表：
{cluster_info}

你的任务是从中找出所有指代【完全相同核心概念】的实体，并将它们分到同一个组中进行合并。

【合并准则（非常重要）】：
1. ✅ 必须合并的情况（包含语法差异与深度语义等价）：
   - 基础语法差异：大小写、单复数、无意义的连字符 (如 "x ray" 与 "x-ray")、拼写错误。
   - 化学式与全称等价：(例如 "Al2O3", "alumina", "aluminum oxide" 指代同一物质，必须合并)。
   - 学术缩写与全称等价：(例如 "BCC" 与 "body-centered cubic" 必须合并)。
   - 材料学公认同义词/俗称：(例如 "mild steel" 与 "low-carbon steel" 必须合并)。
   
2. ❌ 严禁合并的情况（材料学语义发生实质改变，即使字面很像）：
   - 不同的物相/晶型：(例如 "TiO2" 与 "rutile TiO2" 不能合并，后者是特定的金红石相；"carbon" 与 "graphene" 不能合并)。
   - 父类与子类/修饰状态：(例如 "titanium alloy" 与 "cast titanium alloy" 不能合并；"steel" 与 "stainless steel" 不能合并)。
   - 形貌/尺度差异：(例如 "gold" 与 "gold nanoparticles" 不能合并，宏观块体与纳米颗粒性质完全不同)。
   - 材料与加工工艺：(例如 "hydroxyapatite" 和 "deposition of hydroxyapatite" 绝不能合并)。

【输出要求】：
请你先输出一步 "reasoning" (分析过程)。在分析时，请像材料科学家一样，指出哪些词是化学式与俗称的对应，哪些是缩写，哪些因为晶型、形貌、修饰词不同而不能合并。
然后再输出 "merge_groups" 数组。如果某几个实体语义等价，请将它们归为一组，并提取/生成一个最规范的名字（首选全称或最通用的学名）。

严格返回以下 JSON 格式：
{{
  "reasoning": "简要的分析过程...",
  "merge_groups": [
    {{
      "canonical_name": "选择或生成的最规范名称",
      "entities_to_merge": ["必须填入具体的节点 Name 字符串，绝不能填 '实体1'、'实体2' 这种代号！", "具体的 Name 字符串2..."] 
    }}
  ]
}}
如果完全没有可以合并的，"merge_groups" 留空 []。
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "system", "content": "你是一个严谨的材料学术语对齐助手，只输出标准 JSON。"},
                          {"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw_content = response.choices[0].message.content
            
            # 使用正则匹配大括号包围的 JSON 内容
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                decision = json.loads(json_str)
                
                # 打印大模型的材料学思维链
                if "reasoning" in decision and decision["reasoning"]:
                    self._log_action(f"   💡 [LLM 语义分析过程]: {decision['reasoning']}")
                    
                return decision
            else:
                return {"merge_groups": []}
                
        except Exception as e:
            self._log_action(f"   ❌ [LLM 警告] 推断或解析过程失败: {e}")
            return {"merge_groups": []}

    def run_alignment_cycle(self, threshold=90):
        """执行一轮完整的对齐流程"""
        entities = self.get_all_entities()
        names = [e['name'] for e in entities]
        processed_names = set() # 已经被物理合并掉（消失）的节点
        evaluated_clusters = set() # 记录已经评估过的聚类组合，防止无限重复循环
        
        cycle_merged_count = 0

        print(f"开始本轮对齐，共有 {len(entities)} 个初始节点...")

        for entity in entities:
            if entity['name'] in processed_names:
                continue
            
            # 1. 基于编辑距离寻找相似候选
            # 核心优化：只在尚未被合并的“存活”节点中寻找
            available_names = [n for n in names if n not in processed_names]
            matches = process.extract(
                entity['name'], 
                available_names, 
                scorer=fuzz.WRatio, 
                score_cutoff=threshold, 
                limit=15 # 适当放宽 limit，让大集群暴露更完整
            )
            
            if len(matches) <= 1:
                continue
            
            # 准备聚类数据
            current_cluster_names = [m[0] for m in matches]
            # 将当前这组名字排序并哈希。如果之前大模型已经看过了完全一样的一组词，直接跳过！
            cluster_signature = tuple(sorted(current_cluster_names))
            if cluster_signature in evaluated_clusters:
                continue
            evaluated_clusters.add(cluster_signature)
            
            current_cluster_data = [e for e in entities if e['name'] in current_cluster_names]
            
            # 2. 双向打印推断节点信息
            self._log_action("\n🧠 [推断节点信息] 准备发送给大模型进行语义校验的候选节点：")
            for node in current_cluster_data:
                self._log_action(f"   ├─ Name: [{node['name']}]  |  Type (Labels): {node['labels']}")
                
            decision = self._ask_llm_for_canonical_form(current_cluster_data)
            
            # 3. 解析大模型返回的多个合并分组
            merge_groups = decision.get("merge_groups", [])
            # 提取大模型的思维链推理过程
            reasoning_text = decision.get("reasoning", "")

            if not merge_groups:
                self._log_action("   ❌ [跳过] 大模型判定：本组内没有可以合并的变体，均属于独立概念。")
                continue
            
            # 遍历大模型挑出的每一个可以合并的子集
            for group in merge_groups:
                entities_to_merge = group.get("entities_to_merge", [])
                canonical_name = group.get("canonical_name", "")
                
                # 安全校验：防止大模型幻觉输出了不在当前聚类里的实体
                valid_entities_to_merge = [n for n in entities_to_merge if n in current_cluster_names]
                
                if len(valid_entities_to_merge) > 1:
                    actual_merged = self._execute_merge(valid_entities_to_merge, canonical_name, reasoning_text)
                    if actual_merged:
                        cycle_merged_count += actual_merged
                    # 标记合并掉的节点，防止在后续循环中再作为查询目标W
                    for n in valid_entities_to_merge:
                        processed_names.add(n)
                elif len(entities_to_merge) > 1:
                    self._log_action(f"   ⚠️ [格式拦截] 大模型未按要求输出具体名称（它输出了: {entities_to_merge}），无法匹配，跳过本组。")
        return cycle_merged_count

    def _execute_merge(self, names_to_merge, canonical_name, reasoning=""):
        """
        执行物理合并：
        1. 优先级判定：标准术语 (source_doc='origin') > 文献术语(Term) > 碎片(Other)。
        2. 名称保护：要么继承标准术语原名，要么采用 LLM 生成的规范名。
        3. 标签控制：合并后只要含 Term，最终强转为纯 Term。
        """
        # 修改查询，提取 source_doc 属性
        check_query = """
        MATCH (n) 
        WHERE n.name IN $names AND (n:Term OR n:Other) AND NOT n:Classification
        RETURN id(n) AS id, n.name AS name, labels(n) AS labels, n.source_doc AS source_doc
        """
        
        with self.driver.session() as session:
            records = session.run(check_query, names=names_to_merge).data()
            
            if not records:
                return
            
            # --- 1. 优先级血统判定 ---
            def get_priority(rec):
                # 只有 source_doc 为 origin 且标签包含 Term，才是真正的“标准节点”
                if 'Term' in rec['labels'] and rec['source_doc'] == 'origin':
                    return 0 
                if 'Term' in rec['labels']:
                    return 1 # 已晋升的普通文献术语
                return 2     # 末端：Other 节点

            sorted_records = sorted(records, key=get_priority)
            
            all_original_names = [r['name'] for r in sorted_records]
            entities_with_labels = [f"{r['name']} ({', '.join(r['labels'])})" for r in sorted_records]

            # ================== 核心更新：柔性互斥安全锁 ==================
            # 统计有多少个 source_doc == 'origin' 的权威标准节点
            origin_nodes = [r for r in sorted_records if r.get('source_doc') == 'origin']
            if len(origin_nodes) > 1:
                # 提取除了第一个（作为最高优先级主节点）之外的其他标准节点
                protected_origins = origin_nodes[1:]
                protected_ids = set(r['id'] for r in protected_origins)
                protected_names = [r['name'] for r in protected_origins]
                
                self._log_action(f"  [安全拦截] 🛡️ 发现大模型试图合并多个标准节点！已将 {protected_names} 从名单中剔除以作保护。")
                
                # 从待合并列表中过滤掉受保护的节点
                sorted_records = [r for r in sorted_records if r['id'] not in protected_ids]
                
                # 过滤后，如果该组只剩下 1 个（或 0 个）节点了，则说明无需合并，直接跳过
                if len(sorted_records) <= 1:
                    self._log_action("  [安全拦截] 剔除冲突的标准节点后，无其余可合并的子节点，本次合并取消。\n")
                    return 0
            # ==============================================================
            
            target_record = sorted_records[0]
            target_id = target_record['id']
            
            ordered_ids = [target_id] + [r['id'] for r in sorted_records[1:]]
            has_term = any('Term' in r['labels'] for r in sorted_records)
            all_original_names = [r['name'] for r in sorted_records]
            
            # --- 2. 【核心更新】：Python 端提前决断最终名称 ---
            # 如果排在第一位的主节点是“标准节点”，绝对不改名；否则听 LLM 的
            if 'Term' in target_record['labels'] and target_record['source_doc'] == 'origin':
                final_decided_name = target_record['name']
            else:
                final_decided_name = canonical_name

            # 使用双向日志记录合并总览
            self._log_action(f"  [物理合并] 准备合并 {len(ordered_ids)} 个节点... (最终选用名称: [{final_decided_name}])")
            self._log_action(f"  [参与合并的具体节点信息]：")
            
            # 遍历打印所有参与合并的节点具体名称和类型
            for rec in sorted_records:
                self._log_action(f"     ├─ Name: [{rec['name']}]  |  Type (Labels): {rec['labels']}")
            
            # --- 3. 执行合并与打标 ---
            merge_query = """
            MATCH (n) WHERE id(n) IN $ordered_ids
            WITH collect(n) AS nodes
            // properties: 'combine' 会把不同值合并为数组，相同值去重
            CALL apoc.refactor.mergeNodes(nodes, {
                properties: 'combine',
                mergeRels: true
            }) YIELD node
            
            // 标签控制：如果涉事节点中有 Term，则强制纯粹化
            FOREACH (ignoreMe IN CASE WHEN $has_term THEN [1] ELSE [] END |
                SET node:Term
                REMOVE node:Other
            )
            
            // 图谱内溯源标记
            SET node.is_merged = true
            
            SET node.entities_before_merge = $entities_with_labels

            SET node.merged_from_history = CASE 
                WHEN node.merged_from_history IS NULL THEN $all_names 
                ELSE apoc.coll.toSet(node.merged_from_history + $all_names) 
            END
            
            // 强制覆盖名称（防止 combine 把名字变成数组）
            SET node.name = $final_decided_name
            
            RETURN node.name AS final_name, labels(node) AS final_labels
            """
            result = session.run(merge_query, 
                        ordered_ids=ordered_ids, 
                        has_term=has_term, 
                        final_decided_name=final_decided_name,
                        all_names=all_original_names,
                        entities_with_labels=entities_with_labels).single()  # <--- 加上这一行参数
            # 4. 写入本地文件日志与终端双向打印最终结果
            self._log_merge(all_original_names, final_decided_name, reasoning)
            if result:
                self._log_action(f"  [合并结果] 🌟 最终实体形态 -> Name: [{result['final_name']}]  |  Type (Labels): {result['final_labels']}\n")
            
            return len(ordered_ids)

# ================= 调用入口 =================
if __name__ == "__main__":
    aligner = EntityAligner(
        "bolt://localhost:7687", "neo4j", "LPZ12256zpl",
        "sk-fd64f2be977547b4a10ee6a3316021b3", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    total_merged_nodes = 0 # 【新增】全局计数器

    # 4. 多轮迭代优化 [建议执行 3 次]
    for i in range(3):
        print(f"\n--- 第 {i+1} 轮迭代优化开始 ---")
        # 【修改】接收每轮的合并数量
        merged_in_cycle = aligner.run_alignment_cycle(threshold=92) 
        if merged_in_cycle:
            total_merged_nodes += merged_in_cycle
    
    # 【新增】所有迭代完成后，双向打印（终端+日志文件）最终统计结果
    summary_msg = (
        f"\n==================================================\n"
        f"📊 实体对齐任务全部结束！\n"
        f"   总计参与物理合并的实体节点数量: {total_merged_nodes} 个\n"
        f"=================================================="
    )
    aligner._log_action(summary_msg)
    
    aligner.close()