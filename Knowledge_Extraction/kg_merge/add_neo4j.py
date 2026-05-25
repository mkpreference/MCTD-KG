import json
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from openai import OpenAI
import datetime
import os

class MCTDGraphFuser:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _log_promotion(self, message, log_file="/root/files_lpz/ra/实验/literature/extract/merge/log_addliterature/promotion_details_log.txt"):
        """将晋升协议的每一步细节记录到本地文件，并同步打印到控制台"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(message)  # 保持控制台输出
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_message + "\n")

    def _log_anomaly(self, message, log_file="/root/files_lpz/ra/实验/literature/extract/merge/log_addliterature/triplet_anomaly_log.txt"):
        """将幻觉修复和残缺丢弃的记录追加到本地文件中"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 以追加模式(a)打开日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    def parse_source_key(self, key_string):
        """解析溯源信息"""
        parts = key_string.split("__")
        category = parts[0] if len(parts) > 0 else "Unknown_Category"
        source_doc = parts[1] if len(parts) > 1 else "Unknown_Doc"
        chunk_id = parts[2] if len(parts) > 2 else "Unknown_Chunk"
        return category, source_doc, chunk_id

    def fuse_literature_json(self, json_file_path):
        """读取合并后的 JSON 并将其注入 Neo4j"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for source_key, triplets in data.items():
            category, source_doc, chunk_id = self.parse_source_key(source_key)
            print(f"\n==================================================")
            print(f"📄 开始融合文献: {source_doc} (分类: {category})")
            print(f"==================================================")

            if not triplets:
                continue

            for t in triplets:
                if not isinstance(t, dict):
                    continue
                try:
                    # 【路径 1】：完全标准的“关系三元组”
                    if "relation" in t and "head" in t and "tail" in t:
                        print(f"  [写入关系] 🔗 ({t['head']}) -[{t['relation']}]-> ({t['tail']})")
                        self._merge_relation(t["head"], t["relation"], t["tail"],
                                             category, source_doc, chunk_id)
                                             
                    # 【路径 2】：完全标准的“属性三元组”
                    elif "attribute" in t and "entity" in t and "value" in t:
                        print(f"  [写入属性] 📝 ({t['entity']}) -[{t['attribute']}]-> ({t['value']})")
                        self._merge_attribute(t["entity"], t["attribute"], t["value"],
                                              category, source_doc, chunk_id)
                                              
                    # 【路径 3】：容错/幻觉抢救 - 关系三元组
                    elif "relation" in t:
                        head_name = t.get("head") or t.get("entity")
                        tail_name = t.get("tail") or t.get("value")
                        
                        if head_name and tail_name:
                            msg = f"[幻觉修复-关系] 文献: {source_doc} | Chunk: {chunk_id} | 原始异常数据: {t} ---> 修复为: ({head_name}) -[{t['relation']}]-> ({tail_name})"
                            print(f"  ⚠️ {msg}")
                            self._log_anomaly(msg) # 写入文件
                            
                            self._merge_relation(head_name, t["relation"], tail_name,
                                                 category, source_doc, chunk_id)
                        else:
                            msg = f"[残缺丢弃-关系] 文献: {source_doc} | Chunk: {chunk_id} | 严重残缺数据: {t}"
                            print(f"  ❌ {msg}")
                            self._log_anomaly(msg) # 写入文件

                    # 【路径 4】：容错/幻觉抢救 - 属性三元组
                    elif "attribute" in t:
                        entity_name = t.get("entity") or t.get("head")
                        value_name = t.get("value") or t.get("tail")
                        
                        if entity_name and value_name:
                            msg = f"[幻觉修复-属性] 文献: {source_doc} | Chunk: {chunk_id} | 原始异常数据: {t} ---> 修复为: ({entity_name}) -[{t['attribute']}]-> ({value_name})"
                            print(f"  ⚠️ {msg}")
                            self._log_anomaly(msg) # 写入文件
                            
                            self._merge_attribute(entity_name, t["attribute"], value_name,
                                                  category, source_doc, chunk_id)
                        else:
                            msg = f"[残缺丢弃-属性] 文献: {source_doc} | Chunk: {chunk_id} | 严重残缺数据: {t}"
                            print(f"  ❌ {msg}")
                            self._log_anomaly(msg) # 写入文件
                            
                    # 【路径 5】：彻底不可读的垃圾数据
                    else:
                        msg = f"[彻底丢弃-未知格式] 文献: {source_doc} | Chunk: {chunk_id} | 垃圾数据: {t}"
                        print(f"  ❌ {msg}")
                        self._log_anomaly(msg) # 写入文件
                        continue
                        
                except Exception as e:
                    error_msg = f"[代码运行报错] 文献: {source_doc} | 错误信息: {e} | 引发错误的数据: {t}"
                    print(f"  [报错跳过] {error_msg}")
                    self._log_anomaly(error_msg) # 将异常栈也记录下来

    def _merge_relation(self, head, relation, tail, category, source_doc, chunk_id):
        rel_clean = str(relation).replace('`', '').strip()
        
        # 【机制更新】：去掉了 MERGE 里的 :Entity 标签限制
        # 直接通过 {name: $head} 全局匹配。如果不存在，则 ON CREATE 只赋予 :Other 标签！
        query = f"""
        MERGE (h {{name: $head}})
        ON CREATE SET h:Other, h.source_doc = 'literature', h.source_chunk = $source_doc
        SET h.inferred_category = $category

        MERGE (t {{name: $tail}})
        ON CREATE SET t:Other, t.source_doc = 'literature', t.source_chunk = $source_doc
        SET t.inferred_category = $category

        WITH h, t
        MERGE (h)-[rel:`{rel_clean}`]->(t)
        ON CREATE SET rel.source_chunk = $source_doc, rel.source_doc = 'literature'
        RETURN count(rel) AS cnt
        """
        with self.driver.session() as session:
            session.run(query, head=head, tail=tail, category=category, 
                        source_doc=source_doc, chunk_id=chunk_id)

    def _merge_attribute(self, entity_name, attribute_name, value, category, source_doc, chunk_id):
        attr_clean = str(attribute_name).replace('`', '').strip()
        attr_source_key = attr_clean + "_source"
        
        # 【机制更新】：同理，去掉了 :Entity 标签
        query = f"""
        MERGE (e {{name: $entity}})
        ON CREATE SET e:Other, e.source_doc = 'literature', e.source_chunk = $source_doc
        SET e.source_category = $category

        SET e.`{attr_clean}` = $value
        SET e.`{attr_source_key}` = $source_doc
        RETURN e
        """
        with self.driver.session() as session:
            session.run(query, entity=entity_name, value=value,
                        category=category, source_doc=source_doc)

    # ==================== 请将以下三个方法添加/替换到您的类中 ====================
    def _evaluate_term_validity(self, llm_client, name, context_info):
        """
        让 LLM 评估该节点是否为有效的材料专业领域术语。
        返回 (is_valid: bool, reason: str)
        """
        prompt = f"""
        你是一位材料科学领域的专家。现在有一个从工具书或学术文献中抽取的候选节点，需要你判断它是否具备成为正式“专业术语（Term）”的资格。
        候选词：“{name}”  
        上下文关联：{context_info}
        
        判断标准：
        - 该节点应代表材料科学领域内具有明确、稳定、公认含义的专业概念；
        - 不应是泛化词汇、普通名词、过程描述、设备名称、人名、机构名或临时组合短语；
        - 应在专业文献、教材或标准中有明确定义或广泛使用。
        任务：请严格根据上述标准，判断该候选节点是否应被升级为“Term”节点。 
        请严格按以下 JSON 格式输出，不要任何额外内容：
{{
    "is_valid": true 或 false,
    "reason": "不超过30字的简要理由"
}}
        """
        # 这里替换为你实际调用 LLM 的代码，并解析 JSON
        try:
            response = llm_client.chat(prompt) # 伪代码
            import json
            # 假设大模型返回了合法的 json 字符串
            result = json.loads(response)
            return result.get("is_valid", False), result.get("reason", "未提供理由")
        except Exception as e:
            print(f"LLM 评估异常: {e}")
            # 发生异常时，默认放行（或者默认拦截，取决于你的严格程度）
            return True, "评估过程发生异常，默认放行"
        
    def _get_taxonomy_tree(self, top_category):
        """获取顶层分类下的所有最底层子分类路径，供 LLM 参考"""
        # MATCH 查询所有路径，WHERE 拦截掉所有中间节点，只保留没有更底层子类的叶子节点
        query = """
        MATCH path = (root:Classification {name: $top_category})<-[:subclassOf*0..]-(leaf:Classification)
        WHERE NOT (leaf)<-[:subclassOf]-(:Classification)
        RETURN [n IN nodes(path) | n.name] AS category_path
        """
        with self.driver.session() as session:
            result = session.run(query, top_category=top_category)
            # 将列表拼接成可视化的路径，由于只查了叶子节点，这里全是直达底层的完整路径
            paths = [ " -> ".join(record["category_path"]) for record in result ]
            return paths if paths else [top_category]
    
    def _ask_llm_for_top_category(self, llm_client, entity_name, context_info):
        """当缺失顶层分类时，调用 LLM 从 11 个核心大类中预测归属"""
        top_categories = [
            "Fundamental_of_Material_Science", "Metals", "Inorganic_non-metallic_materials",
            "Organic_polymer_materials", "Composites", "Information_materials",
            "Energy_Materials", "Biomedical Materials", "Natural_materials_and_products",
            "Special_functional_materials", "Nanomaterials"
        ]
        prompt = f"""
        你是一个材料科学专家。现在有一个术语缺乏顶层分类，需要你将其归入最合适的材料大类中。
        
        【术语名称】: {entity_name}
        【在图谱中的上下文关联】: {', '.join(context_info)}
        
        【候选顶层分类列表】:
        {json.dumps(top_categories, indent=2, ensure_ascii=False)}

        任务：请根据上下文，为该术语选择唯一一个最合适的候选顶层分类。
        要求：只需输出该分类的【名称字符串】（必须完全匹配列表中的拼写），不要输出其他任何解释和标点。
        """
        try:
            response = llm_client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {"role": "system", "content": "你是一个严格的材料分类助手，只输出目标分类节点名称。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [LLM调用失败] 无法预测顶层分类: {e}")
            return None
        
    def _ask_llm_for_exact_category(self, llm_client, entity_name, context_info, candidate_paths):
        """调用 LLM 为晋升节点选择最细粒度的子分类"""
        prompt = f"""
        你是一个材料科学知识图谱专家。现在有一个新术语需要被挂载到分类树上。
        
        【新术语名称】: {entity_name}
        【在文献中的上下文/关联信息】: {', '.join(context_info)}

        以下是目前图谱中该顶层大类下的可用分类路径（从粗到细）：
        {json.dumps(candidate_paths, indent=2, ensure_ascii=False)}

        任务：请根据上下文，为该术语选择一个最精确的末端子分类。
        要求：只需输出该分类的【节点名称】（例如：Metal Implants），不要输出完整路径，不要输出任何额外解释。如果提供的路径中没有合适的子分类，请直接输出顶层大类名称。
        """
        try:
            # 兼容您在 source: 2 中的 dashscope/openai 客户端调用方式
            response = llm_client.chat.completions.create(
                model="qwen-max",
                messages=[
                    {"role": "system", "content": "你是一个只输出目标分类节点名称的实体对齐助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1 # 低温度保证输出稳定性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [LLM调用失败] 回退到原始分类: {e}")
            return None
#     #- 关系数为 0 的节点有 439 个
#   - 关系数为 1 的节点有 13586 个
#   - 关系数为 2 的节点有 25663 个
#   - 关系数为 3 的节点有 5989 个
#   - 关系数为 4 的节点有 2977 个
#   - 关系数为 5 的节点有 1957 个
#   - 关系数为 6 的节点有 1370 个
#   - 关系数为 7 的节点有 943 个
#   - 关系数为 8 的节点有 698 个
#   - 关系数为 9 的节点有 541 个
#   - 关系数为 10 的节点有 427 个
#   - 关系数为 11 的节点有 346 个
#   - 关系数为 12 的节点有 229 个
#   - 关系数为 13 的节点有 187 个
#   - 关系数为 14 的节点有 167 个
#   - 关系数为 15 的节点有 124 个
#   - 关系数为 16 的节点有 119 个
#   - 关系数为 17 的节点有 82 个
#   - 关系数为 18 的节点有 88 个
#   - 关系数为 19 的节点有 62 个
#   - 关系数为 20 的节点有 60 个
    #初步选定关系数为6
    def execute_dynamic_node_promotion(self, llm_client, degree_threshold=8):
        """【全新重构】：结合 LLM 智能寻找最深层子分类的晋升协议"""
        self._log_promotion(f"\n🚀 开始执行智能动态晋升协议... (触发阈值: 连接边数 >= {degree_threshold})")
        
        # 1. 查询所有符合晋升条件的节点，排除已经评估过且被拒绝的节点 (n.term_rejected IS NULL)
        fetch_query = """
        MATCH (n:Other)
        WHERE n.term_rejected IS NULL
        // 【修改此处】：使用兼容 Neo4j 4.4 的模式推导写法，既能消除警告，又不会报错
        WITH n, size([(n)-[r]-() | r]) AS degree
        WHERE degree >= $threshold
        
        OPTIONAL MATCH (n)-[r]-(neighbor)
        WITH n, degree, collect(type(r) + ' ' + neighbor.name)[0..5] AS context_info
        
        RETURN id(n) AS node_id, n.name AS name, n.source_category AS top_category, degree, context_info
        """
        
        promoted_count = 0
        rejected_count = 0
        promoted_names = []

        with self.driver.session() as session:
            candidates = session.run(fetch_query, threshold=degree_threshold).data()
            
            for cand in candidates:
                node_id = cand["node_id"]
                name = cand["name"]
                top_category = cand["top_category"]
                degree = cand["degree"]
                context_info = cand["context_info"]
                target_category = top_category
                
                self._log_promotion(f"\n--------------------------------------------------")
                self._log_promotion(f"🔍 正在处理候选节点: '{name}' (连接度: {degree})")
                self._log_promotion(f"   上下文: {context_info}")

                # ================= 新增：LLM 价值评估关卡 =================
                print(f"  [价值评估] 正在请 LLM 鉴定 '{name}' 是否为材料专业领域的有效术语...")
                is_valid, reason = self._evaluate_term_validity(llm_client, name, context_info)

                if not is_valid:
                    self._log_promotion(f"  ❌ [拒绝晋升] '{name}' 被判定为非专业术语。理由: {reason}")
                    # 在图谱中打上拒绝标记，防止下次再被查出来重复调用 LLM
                    reject_query = "MATCH (n) WHERE id(n) = $node_id SET n.term_rejected = true, n.reject_reason = $reason"
                    session.run(reject_query, node_id=node_id, reason=reason)
                    rejected_count += 1
                    continue # 直接跳过当前循环，不执行后续的预测和分类操作
                
                self._log_promotion(f"  ✅ [评估通过] '{name}' 是有效术语，继续晋升流程。")
                # =========================================================

                # 2. 如果缺失顶层分类，先让 LLM 预测一个宏观大类
                if not top_category or top_category == 'Unknown_Category':
                    self._log_promotion(f"  [缺失分类] 正在呼叫 LLM 预测顶层大类...")
                    guessed_top = self._ask_llm_for_top_category(llm_client, name, context_info)
                    
                    if guessed_top:
                        top_category = guessed_top
                        target_category = guessed_top
                        self._log_promotion(f"  [预测成功] '{name}' 被划归至顶层大类: [{top_category}]")
                    else:
                        self._log_promotion(f"  [预测失败] 跳过子分类寻找。")
               
                # 3. 拉取分类树并请求 LLM 进行精准定位（向下细化）
                if top_category and top_category != 'Unknown_Category':
                    paths = self._get_taxonomy_tree(top_category)
                    if len(paths) > 1:
                        self._log_promotion(f"  [LLM思考中] 正在为 '{name}' 寻找 '{top_category}' 下的最佳子分类...")
                        llm_choice = self._ask_llm_for_exact_category(llm_client, name, context_info, paths)
                        if llm_choice:
                            target_category = llm_choice
                            
                # 4. 执行图谱更新
                update_query = """
                MATCH (n) WHERE id(n) = $node_id
                REMOVE n:Other
                SET n:Term, 
                    n.is_promoted = true, 
                    n.promotion_degree = $degree,
                    n.promotion_date = datetime()

                WITH n
                OPTIONAL MATCH (n)-[r:subclassOf]-(h:Head {name: 'Other'})
                DELETE r

                WITH n
                FOREACH (ignoreMe IN CASE WHEN $target_category IS NOT NULL AND $target_category <> 'Unknown_Category' THEN [1] ELSE [] END |
                    MERGE (c:Classification {name: $target_category})
                    MERGE (n)-[:isAKindOf]->(c)
                )
                """
                session.run(update_query, node_id=node_id, degree=degree, target_category=target_category)

                promoted_count += 1
                promoted_names.append(f"{name}  --->  [{target_category}]")
                self._log_promotion(f"  🎉 成功: '{name}' 晋升并挂载至 [{target_category}]")

        # ================= 写入最终统计结果至日志 =================
        self._log_promotion(f"\n📊 最终晋升统计结果：")
        self._log_promotion(f"  - 成功从 (Other) 蜕变为 (Term) 的节点总数：{promoted_count} 个")
        self._log_promotion(f"  - 被 LLM 判定为无效并拒绝的节点总数：{rejected_count} 个")
        
        if promoted_count > 0:
            self._log_promotion("🌟 成功晋升的节点具体名单及目标分类：")
            for i, info in enumerate(promoted_names, 1):
                self._log_promotion(f"   {i}. {info}")
        self._log_promotion(f"================ 晋升协议执行结束 ================\n")

# ================= 运行入口 =================
if __name__ == "__main__":
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "LPZ12256zpl" # 密码保持为您上次提供的

    fuser = MCTDGraphFuser(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    llm_client = OpenAI(
        api_key="sk-fd64f2be977547b4a10ee6a3316021b3",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # # 1. 融合文献抽取的JSON数据
    # fuser.fuse_literature_json(
    #     "/root/files_lpz/ra/实验/literature/extract/merge/parsed_results/all_parsed.json"
    # )

    # 2. 执行晋升协议（阈值选取为8）
    fuser.execute_dynamic_node_promotion(llm_client=llm_client,degree_threshold=8)

    fuser.close()

    '''
    ==================================================
📄 开始融合文献: 1-s2.0-S0925838826010510-main_chunks (分类: Energy_Materials)
==================================================
  [写入关系] 🔗 (Cr2O3) -[isUsedFor]-> (filling pores)
  [写入关系] 🔗 (ZrCr2 intermetallic layer) -[isPartOf]-> (Cr-coated Zr alloy cladding)
  [写入关系] 🔗 (failure of Cr coating) -[isDependentOn]-> (eutectic liquid phase)
  [写入关系] 🔗 (Mo interlayer) -[isUsedFor]-> (suppressing interdiffusion of Cr and Zr)
  [写入关系] 🔗 (Cr-Mo-Zr eutectic liquid phase) -[isDerivedFrom]-> (Zr-Cr interdiffusion)
  [写入关系] 🔗 (ZrO2 layer) -[isPartOf]-> (structure between Zr-C layer and Zr substrate)
Traceback (most recent call last):
  File "/root/files_lpz/ra/实验/literature/extract/merge/add_neo4j.py", line 317, in <module>
    fuser.fuse_literature_json(
  File "/root/files_lpz/ra/实验/literature/extract/merge/add_neo4j.py", line 44, in fuse_literature_json
    print(f"  [写入属性] 📝 ({t['entity']}) -[{t['attribute']}]-> ({t['value']})")
KeyError: 'entity'


==================================================
📄 开始融合文献: structural-and-microstructural-characterization-of-fission-p (分类: Energy_Materials)
==================================================
  [写入关系] 🔗 (gray phases) -[isPartOf]-> ((U,Pu,FP)O2 matrix)
  [写入关系] 🔗 (BaZrO3) -[isDerivedFrom]-> (BaO)
  [写入关系] 🔗 (BaZrO3) -[isDerivedFrom]-> (ZrO2)
  [写入关系] 🔗 (Ba−Zr inclusions) -[isMadeOf]-> (Ba)
  [写入关系] 🔗 (Ba−Zr inclusions) -[isMadeOf]-> (Zr)
  [写入关系] 🔗 (FCC phase) -[isPartOf]-> (sample SM)
  [写入关系] 🔗 (HCP phase) -[isPartOf]-> (sample SM)
  [写入关系] 🔗 (BCC#1 phase) -[isPartOf]-> (sample SM)
  [写入关系] 🔗 (BCC#2 phase) -[isPartOf]-> (sample SM)
  [写入关系] 🔗 (PC phase) -[isPartOf]-> (sample SMB)
  [写入关系] 🔗 (HCP phase) -[isPartOf]-> (sample SMB)
  [写入关系] 🔗 (BCC#1 phase) -[isPartOf]-> (sample SMB)
  [写入关系] 🔗 (BCC#2 phase) -[isPartOf]-> (sample SMB)
  [写入关系] 🔗 (gray phases) -[isPartOf]-> (UO2-SIMfuel)
Traceback (most recent call last):
  File "/root/files_lpz/ra/实验/literature/extract/merge/add_neo4j.py", line 317, in <module>
    fuser.fuse_literature_json(
  File "/root/files_lpz/ra/实验/literature/extract/merge/add_neo4j.py", line 44, in fuse_literature_json
    print(f"  [写入属性] 📝 ({t['entity']}) -[{t['attribute']}]-> ({t['value']})")
KeyError: 'entity'
    '''