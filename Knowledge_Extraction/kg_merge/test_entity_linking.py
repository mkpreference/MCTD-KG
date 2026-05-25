import json
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz
from openai import OpenAI
import datetime

class FuzzMergeTester:
    def __init__(self, uri, user, password, api_key, base_url):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def close(self):
        self.driver.close()

    def get_all_entities(self):
        """获取所有节点，【新增】带上 source_doc 字段用于标准节点判定"""
        query = """
        MATCH (n)
        WHERE n:Term OR n:Other
        OPTIONAL MATCH (n)-[r]-(neighbor)
        RETURN id(n) AS node_id, n.name AS name, labels(n) AS labels, 
               n.source_category AS category, n.source_doc AS source_doc,
               collect(DISTINCT neighbor.name)[0..3] AS neighbors,
               properties(n) AS props
        """
        with self.driver.session() as session:
            return session.run(query).data()

    def _print_snapshot(self, title, entities):
        """打印节点的详细状态快照"""
        print(f"\n{title}")
        for e in entities:
            print(f"  ├─ 节点名称: {e.get('name')}")
            print(f"  │  ├─ 内部 ID: {e.get('node_id')}")
            print(f"  │  ├─ 标    签: {e.get('labels')}")
            print(f"  │  ├─ 来    源: {e.get('source_doc')}")
            print(f"  │  └─ 属    性: {{k:v for k,v in e.get('props', {{}}).items() if not k.startswith('_') and k != 'uri'}}")
            print("  │")

    def _ask_llm_for_canonical_form(self, cluster_data):
        """调用 LLM 判断聚类中的实体是否可以合并"""
        cluster_info = ""
        for i, item in enumerate(cluster_data):
            cluster_info += f"实体{i}: {item['name']} (分类: {item['category']}, 邻居: {item['neighbors']})\n"

        prompt = f"""
你是一个材料科学实体对齐专家。以下是一组基于语法相似度聚类的实体：
{cluster_info}
任务要求：
1. 识别这组实体中，哪些是真正的【同义词】或【语法变体】。
2. 识别哪些实体虽然字面相似，但【语义不同】。
3. 对于可以合并的实体，给出一个最标准、最规范的名称。
4. 严格按照以下 JSON 格式输出：
{{
  "can_merge": true/false,
  "canonical_name": "最规范的名字",
  "entities_to_merge": ["实体名1", "实体名2"]
}}
如果不能合并，请输出 "can_merge": false。
"""
        try:
            response = self.client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "system", "content": "你是一个严谨的助手。只输出纯 JSON。"},
                          {"role": "user", "content": prompt}],
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content.replace('```json', '').replace('```', '').strip())
        except Exception as e:
            print(f"LLM 推断失败: {e}")
            return {"can_merge": False}

    def _execute_merge(self, names_to_merge, canonical_name):
        """执行极其严格的真实物理合并"""
        check_query = """
        MATCH (n) WHERE n.name IN $names
        RETURN id(n) AS id, n.name AS name, labels(n) AS labels, n.source_doc AS source_doc
        """
        with self.driver.session() as session:
            records = session.run(check_query, names=names_to_merge).data()
            if not records:
                return None
            
            # --- 优先级判定：标准术语绝对优先 ---
            def get_priority(rec):
                if 'Term' in rec['labels'] and rec.get('source_doc') == 'origin': return 0 
                if 'Term' in rec['labels']: return 1 
                return 2     

            sorted_records = sorted(records, key=get_priority)
            target_record = sorted_records[0]
            target_id = target_record['id']
            ordered_ids = [target_id] + [r['id'] for r in sorted_records[1:]]
            
            has_term = any('Term' in r['labels'] for r in sorted_records)
            all_original_names = [r['name'] for r in sorted_records]
            
            # --- 名称保护机制 ---
            if 'Term' in target_record['labels'] and target_record.get('source_doc') == 'origin':
                final_decided_name = target_record['name']
                print(f"  [Action] 🛡️ 触发名称保护！主节点为标准术语，拒绝 LLM 改名，保留原名: {final_decided_name}")
            else:
                final_decided_name = canonical_name
                print(f"  [Action] ✍️ 采用 LLM 生成的规范名称: {final_decided_name}")

            merge_query = """
            MATCH (n) WHERE id(n) IN $ordered_ids
            WITH collect(n) AS nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'combine', mergeRels: true}) YIELD node
            
            FOREACH (ignoreMe IN CASE WHEN $has_term THEN [1] ELSE [] END |
                SET node:Term
                REMOVE node:Other
            )
            
            SET node.is_merged = true
            SET node.merged_from_history = CASE 
                WHEN node.merged_from_history IS NULL THEN $all_names 
                ELSE apoc.coll.toSet(node.merged_from_history + $all_names) 
            END
            SET node.name = $final_decided_name
            RETURN id(node) AS final_id
            """
            result = session.run(merge_query, ordered_ids=ordered_ids, has_term=has_term, 
                                 final_decided_name=final_decided_name, all_names=all_original_names).single()
            return result["final_id"] if result else None

    def _verify_and_print_result(self, target_id):
        """拉取合并后的最终节点状态"""
        verify_query = """
        MATCH (n) WHERE id(n) = $id
        RETURN id(n) AS node_id, n.name AS name, labels(n) AS labels, n.source_doc AS source_doc, properties(n) AS props
        """
        with self.driver.session() as session:
            r = session.run(verify_query, id=target_id).single()
            if r:
                self._print_snapshot("🌟 [快照] 合并后的最终实体形态 (Final State):", [dict(r)])

    def run_targeted_test(self, threshold=90):
        """执行定向查找与测试"""
        entities = self.get_all_entities()
        names = [e['name'] for e in entities]
        processed_names = set()
        
        tested_term_other = False
        tested_other_other = False
        
        print(f"🔍 开始从 {len(entities)} 个节点中扫描测试用例 (编辑距离 > {threshold})...")

        for entity in entities:
            if tested_term_other and tested_other_other:
                print("\n✅ 两组测试用例均已完成！测试脚本安全退出。")
                break
                
            if entity['name'] in processed_names:
                continue
            
            # 1. Rapidfuzz 寻找相似实体
            matches = process.extract(entity['name'], names, scorer=fuzz.WRatio, score_cutoff=threshold, limit=10)
            current_cluster_names = list(set([m[0] for m in matches]))
            
            if len(current_cluster_names) <= 1:
                continue
                
            current_cluster_data = [e for e in entities if e['name'] in current_cluster_names]
            
            # 2. 判断当前聚类的成分
            labels_in_cluster = [e['labels'] for e in current_cluster_data]
            has_term = any('Term' in labels for labels in labels_in_cluster)
            has_other = any('Other' in labels for labels in labels_in_cluster)
            
            case_type = None
            if has_term and has_other and not tested_term_other:
                case_type = "Term + Other"
            elif not has_term and has_other and not tested_other_other:
                case_type = "Other + Other"
                
            if not case_type:
                continue # 不符合我们正在寻找的测试类型，继续找
                
            print(f"\n==================================================")
            print(f"🎯 捕获到测试用例: 【{case_type}】 (候选实体: {current_cluster_names})")
            print(f"==================================================")
            
            # 3. 呼叫大模型验证
            print("🧠 正在呼叫大模型进行语义校验...")
            decision = self._ask_llm_for_canonical_form(current_cluster_data)
            
            if decision.get("can_merge") and len(decision.get("entities_to_merge", [])) > 1:
                print(f"✅ 大模型同意合并！(规范名称建议: {decision.get('canonical_name')})")
                
                # 提取真正要合并的节点详细数据
                nodes_to_merge_data = [e for e in current_cluster_data if e['name'] in decision["entities_to_merge"]]
                
                # 打印合并前快照
                self._print_snapshot("🔻 [快照] 合并前的独立节点状态:", nodes_to_merge_data)
                
                # 执行真实合并
                print("\n⚡ 开始执行底层物理合并与标签洗牌...")
                final_id = self._execute_merge(decision["entities_to_merge"], decision["canonical_name"])
                
                # 打印合并后快照
                if final_id:
                    self._verify_and_print_result(final_id)
                
                # 更新状态
                for n in decision["entities_to_merge"]:
                    processed_names.add(n)
                
                if case_type == "Term + Other":
                    tested_term_other = True
                else:
                    tested_other_other = True
            else:
                print("❌ 大模型拒绝合并，继续寻找下一组...")

# ================= 调用入口 =================
if __name__ == "__main__":
    tester = FuzzMergeTester(
        "bolt://localhost:7687", "neo4j", "LPZ12256zpl",
        "sk-9c64c6fa6e5a4083a8ca0772b6582db0", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    tester.run_targeted_test(threshold=90)
    
    tester.close()