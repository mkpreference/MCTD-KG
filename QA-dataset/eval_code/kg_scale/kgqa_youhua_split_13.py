import json
import re
import numpy as np
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
# ================= 基础配置 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "LPZ12256zpl"

# LLM API 配置 (以通义千问 Qwen 为例，可换为 OpenAI)
API_KEY = "sk-9c64c6fa6e5a4083a8ca0772b6582db0"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"

# 本地轻量级向量模型，用于实现实体的“稠密检索/语义对齐”
EMBEDDING_MODEL_PATH = os.path.abspath("/root/files_lpz/bge-m3") 
# ===========================================

class HybridKBQAPipeline:
    VALID_RELATIONS = [
        "subclassOf", "isAKindOf", "isRelatedTo", "isPartOf", 
        "isUsedFor", "isMadeOf", "isDerivedFrom", "isTestedBy", 
        "isSimilarTo", "isComplementaryTo", "isDependOn", "isAlternativeTo"
    ]
    def __init__(self):
        """初始化数据库连接与语义模型"""
        print("[系统日志] 正在初始化 Neo4j 图数据库连接...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        print("[系统日志] 正在初始化 LLM 客户端...")
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        
        print(f"[系统日志] 正在加载本地向量模型 ({EMBEDDING_MODEL_PATH})...")
        self.embedding_model = SentenceTransformer(
        model_name_or_path=EMBEDDING_MODEL_PATH, 
        device='cuda:1' # 或者 'cpu', 'mps' (Mac)
    )
        
        self.graph_vocabulary = []  # 存储图谱中的所有规范术语名
        self.graph_embeddings = None # 存储对应的向量表示

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def build_graph_vocabulary(self, cache_dir="/root/files_lpz/ra/实验/literature/exp_code/kgqa_13_cache", force_rebuild=False):  # === [MODIFIED] 1/3 消歧实验独立缓存目录 ===
        """
        步骤 1.1 (工程优化版)：从 Neo4j 数据库中动态构建或从本地缓存加载全量图谱词表。
        引入本地缓存机制，避免每次重复计算 Embedding。

        :param cache_dir: 缓存文件存放的目录
        :param force_rebuild: 是否强制忽略缓存，重新从数据库拉取并计算
        """
        import os
        import json
        import numpy as np

        # 定义缓存文件路径
        vocab_cache_path = os.path.join(cache_dir, "graph_vocabulary.json")
        embed_cache_path = os.path.join(cache_dir, "graph_embeddings.npy")

        # ================= 优先尝试从本地加载缓存 =================
        if not force_rebuild and os.path.exists(vocab_cache_path) and os.path.exists(embed_cache_path):
            print(f"[系统日志] 检测到本地缓存目录 '{cache_dir}'，正在直接加载词表与向量...")
            try:
                # 加载词表
                with open(vocab_cache_path, 'r', encoding='utf-8') as f:
                    self.graph_vocabulary = json.load(f)

                # 加载 NumPy 格式的向量矩阵
                self.graph_embeddings = np.load(embed_cache_path)

                print(f"[系统日志] ⚡ 本地缓存加载完毕，共计 {len(self.graph_vocabulary)} 个实体！")
                return # 成功加载，直接结束函数
            except Exception as e:
                print(f"[警告] 缓存加载失败，将重新从图谱拉取数据。错误信息: {e}")

        # ================= 缓存不存在或强制重建，执行原始拉取逻辑 =================
        print("[系统日志] 未使用缓存，正在从 Neo4j 提取 1/3 子图实体名称...")
        # === [MODIFIED] Term/Other 仅取 in_subgraph_1_3=true，Classification 不限 ===
        cypher = """
        MATCH (n)
        WHERE
        n.name IS NOT NULL
        AND
        (
        (n:Term AND n.in_subgraph_1_3 = true)
        OR
        n:Classification
        )
        RETURN DISTINCT n.name AS entity_name
        """
        
        all_entities = []
        with self.driver.session() as session:
            results = session.run(cypher)
            for record in results:
                all_entities.append(record["entity_name"])
        
        self.graph_vocabulary = all_entities
        
        print(f"[系统日志] 提取完成，共计 {len(self.graph_vocabulary)} 个全局实体。")
        print("[系统日志] 正在为全量实体预计算向量表示 (这可能需要 1-2 分钟)...")
        
        self.graph_embeddings = self.embedding_model.encode(self.graph_vocabulary, show_progress_bar=True)
        
        # ================= 将计算结果持久化保存到本地 =================
        print("[系统日志] 正在将词表与向量序列化保存至本地缓存...")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 保存词表 (JSON 格式便于人类可读和排查)
        with open(vocab_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.graph_vocabulary, f, ensure_ascii=False, indent=2)
            
        # 保存向量 (NumPy .npy 格式，加载速度极快)
        np.save(embed_cache_path, self.graph_embeddings)
        
        print(f"[系统日志] 全局向量索引构建并缓存完成。目录: {cache_dir}")
    def extract_intent_via_llm(self, question):
        """
        Step 1: Extract BOTH Entity Mentions and Relationship Intents.
        [Optimized Version]: Uses advanced Prompt Engineering with Schema Mapping Rules 
        instead of hardcoded string matching. It guides the LLM to identify implicit multi-hop relations.
        """
        prompt = f"""
        You are an Information Extraction Expert for a Materials Science Knowledge Graph.
        Your task is to analyze the user's question and extract:
        1. "entities": Extract ALL nouns and noun phrases that could represent nodes in the graph. 
           【CRITICAL】: Do NOT just extract the main subject! You MUST extract ALL intermediate components, peripheral materials, and generic objects mentioned (e.g., 'glass substrate', 'hydrogel membrane', 'porous substance', 'matrix'). These "weak terminologies" are vital for multi-hop graph traversal. Provide multiple variations/granularities.
        2. "relations": ALL the predefined graph relationships required to traverse the graph and answer the question, including implicit structural paths.       
        
        [Predefined Relationships Vocabulary]:
        {json.dumps(self.VALID_RELATIONS)}
        
        [CRITICAL SCHEMA MAPPING RULES]:
        - Rule 1 (Multi-hop Logic): A question often requires multiple relations. For example, finding a term within a specific category that relates to another term requires extracting BOTH the classification relations AND the specific relationship.
        - Rule 2 (Hierarchy & Taxonomy): Whenever the question mentions a term "belonging to", "falling under", or being part of a "classification", "category", or "class", you MUST include BOTH "subclassOf" and "isAKindOf" to bridge the taxonomy tree.
        - Rule 3 (Direct Matching): If the question explicitly mentions a predefined relation (e.g., "isRelatedTo", "isUsedFor"), include it.
        - Rule 4 (Exhaustive Entities): If a question traces a long path (e.g., "A acts as a part of B, integrated into C, tracing to D"), extract ["A", "B", "C", "D"]. This is vital!        
        - Rule 5 (Strict Filtering): If the question is purely about attributes and NONE of the predefined relationships apply, leave "relations" as an empty list []. Do not hallucinate.
        - Rule 6 (Entity Granularity & Variations): Extract concise, atomic entity names by stripping out descriptive adjectives (e.g., extract "BaKPbO3" instead of "lead salt superconductor BaKPbO3"). For complex noun phrases, output MULTIPLE variations of the entity in the list (e.g., ["BaKPbO3", "lead salt superconductor BaKPbO3", "lead salt superconductor"]).
        
        Return ONLY a valid JSON object with the keys "entities" (list of strings) and "relations" (list of strings).
        
        Question: "{question}"
        """
        
        response = self.llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You strictly follow mapping rules and only output JSON objects like {\"entities\": [\"A\"], \"relations\": [\"isRelatedTo\", \"isAKindOf\", \"subclassOf\"]}"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01
        )
        
        output_text = response.choices[0].message.content
        
        try:
            match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if match:
                intent = json.loads(match.group())
                valid_rels = [r for r in intent.get("relations", []) if r in self.VALID_RELATIONS]
                
                # ================= 核心修复升级：强制意图补全 (Intent Imputation) =================
                
                # 规则 1：检查 LLM 是否自己提取了分类关系
                has_classification_intent = "subclassOf" in valid_rels or "isAKindOf" in valid_rels
                
                # 规则 2：关键词硬拦截！检查原始问题中是否含有分类概念的关键词
                question_lower = question.lower()
                has_classification_keywords = any(kw in question_lower for kw in ["分类", "属于", "classification", "category", "parent", "kind of"])
                
                # 只要满足上述任意一点，强制打通图谱的上下级桥梁！
                if has_classification_intent or has_classification_keywords:
                    valid_rels.extend(["subclassOf", "isAKindOf"])
                    
                # 去重
                valid_rels = list(set(valid_rels))
                # =================================================================================
                
                return {
                    "entities": intent.get("entities", []),
                    "relations": valid_rels
                }
            return {"entities": [], "relations": []}
        except Exception as e:
            print(f"[WARNING] Failed to parse LLM intent: {e}")
            return {"entities": [], "relations": []}

    def link_entities(self, mentions, similarity_threshold=0.88,edit_distance_threshold=0.9):
        """
        步骤 2 & 3（极致优化版）：实体链接与严格过滤。
        1. 精确匹配（含符号变体规则）。
        2. 向量语义对齐（在计算向量前，同样引入符号变体进行多路对比，突破高阈值限制）。
        """
        linked_entities = []
        if not mentions:
            return linked_entities
            
        unmatched_mentions = []
        vocab_set = set(self.graph_vocabulary) 
        
        # ================= 1. 第一道关卡：严格的精确匹配 =================
        for mention in mentions:
            if mention in vocab_set:
                linked_entities.append(mention)
                print(f"[实体链接] 精确匹配成功: '{mention}'")
                continue
                
            mention_with_underscore = mention.replace(' ', '_')
            if mention_with_underscore in vocab_set:
                linked_entities.append(mention_with_underscore)
                print(f"[实体链接] 变体精确匹配成功: '{mention}' -> '{mention_with_underscore}'")
                continue
                
            mention_with_space = mention.replace('_', ' ')
            if mention_with_space in vocab_set:
                linked_entities.append(mention_with_space)
                print(f"[实体链接] 变体精确匹配成功: '{mention}' -> '{mention_with_space}'")
                continue
                
            unmatched_mentions.append(mention)
                
        # ================= 2. 第二道关卡：带变体增强的混合对齐 (向量 + 编辑距离) =================
        if unmatched_mentions and self.graph_embeddings is not None:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            from difflib import SequenceMatcher # 引入标准库计算字符串相似度
            
            for mention in unmatched_mentions:
                    # 构建变体集合，确保去重（例如本身就没有空格和下划线的词，集合里就只有1个元素）
                variants = {mention}
                if ' ' in mention:
                    variants.add(mention.replace(' ', '_'))
                if '_' in mention:
                    variants.add(mention.replace('_', ' '))
                variants_list = list(variants)
                
                variant_embeddings = self.embedding_model.encode(variants_list)
                best_overall_score = -1
                best_overall_node = None
                best_string_match_score = -1
                best_string_match_node = None
                
                # 遍历所有图谱实体，计算字面编辑距离 (可选优化：提前通过 Elasticsearch 或 Trie 树过滤)
                for graph_entity in self.graph_vocabulary:
                    ratio = SequenceMatcher(None, mention.lower(), graph_entity.lower()).ratio()
                    if ratio > best_string_match_score:
                        best_string_match_score = ratio
                        best_string_match_node = graph_entity

                # 遍历所有变体的向量，寻找语义最高分
                for i, var_emb in enumerate(variant_embeddings):
                    similarities = cosine_similarity([var_emb], self.graph_embeddings)[0]
                    max_idx = np.argmax(similarities)
                    score = similarities[max_idx]
                    
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_node = self.graph_vocabulary[max_idx]
                
                # 判定逻辑：字面极其相似 OR 语义高度相似
                if best_string_match_score >= edit_distance_threshold:
                     linked_entities.append(best_string_match_node)
                     print(f"[实体链接] 拼写纠错成功: '{mention}' -> '{best_string_match_node}' (编辑距离得分: {best_string_match_score:.4f})")
                elif best_overall_score >= similarity_threshold:
                    linked_entities.append(best_overall_node)
                    print(f"[实体链接] 语义对齐成功: '{mention}' -> '{best_overall_node}' (相似度: {best_overall_score:.4f})")
                else:
                    print(f"[实体过滤] 丢弃无效实体: '{mention}'")
                
        # 去重处理后返回
        return list(set(linked_entities))

    def _filter_attributes_via_llm(self, question, prop_blocks):
        """
        方案 A：属性反向剔除逻辑。
        不再让 LLM 决定“保留什么图谱拓扑”，而是让它仅针对实体的文本属性，
        “删掉绝对无关或产生物理矛盾的属性噪点”。
        """
        if not prop_blocks:
            return []
            
        props_text = "\n".join(prop_blocks)
        prompt = f"""
你是一个材料科学图谱数据清洗专家。以下是从图谱中检索到的关于实体的【属性信息】。
请根据用户的[问题]，剔除那些与问题完全无关、或者与问题中的物理约束产生直接矛盾的属性。

【清洗原则】：
1. 保守过滤：如果你不确定某个属性是否有用，请务必保留它！绝不误删。
2. 物理矛盾：如果问题寻找“高温材料”，而属性显示“熔点极低”，则剔除该属性。
3. 严禁修改：不要修改原始文本，只返回你认为需要保留的属性条目，保持原有的缩进结构。

[实体属性列表]:
{props_text}

[用户问题]:
{question}

请直接输出清洗后的属性列表，不要包含任何解释。
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个严谨的数据清洗接口，只输出清理后的文本。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.01 
            )
            result = response.choices[0].message.content.strip()
            # 按行拆分过滤掉空行，防止格式崩坏
            return [line for line in result.split('\n') if line.strip()]
        except Exception as e:
            print(f"  [LLM 属性过滤报警] 解析失败: {e}，将采用默认全量保留策略兜底。")
            return prop_blocks
        
    def retrieve_subgraph(self, entities, target_relations, question, max_triplets_limit=30):
        """
        Step 4 (终极融合版): Dual-Channel Context Retrieval.
        通道 1: 提取目标节点的全量内部属性。
        通道 2: 提取图谱关联路径 (包含多实体全局最短路径网络，及核心骨架的1跳拓扑辐射)。
        """
        if not entities:
            return "在知识图谱中未匹配到有效实体。"

        properties_context = ""     # 用于存储实体的属性信息
        core_triplets = set()       # 核心骨架路径（绝对保护，不参与剪枝）
        candidate_triplets = set()  # 候选扩展路径（参与向量语义剪枝）
        
        with self.driver.session() as session:
            # ================= 通道一：抓取目标实体的全量内部属性 =================
            # === [MODIFIED] Term/Other 仅查 in_subgraph_1_3=true，Classification 不限 ===
            prop_cypher = """
            MATCH (n)
            WHERE n.name IN $entities 
            AND ((n:Term OR n:Other) AND n.in_subgraph_1_3 = true OR n:Classification)
            RETURN n.name AS e_name, properties(n) AS props
            """
            prop_results = session.run(prop_cypher, entities=entities)
            
            prop_blocks = []
            for record in prop_results:
                e_name = record["e_name"]
                # 将 Neo4j 记录转换为普通字典，防止直接修改底层对象报错
                props = dict(record["props"]) 
                
                # 1. 定义要剔除的元数据黑名单（包含原有的 uri, name，以及图片中的溯源和系统字段）
                exclude_keys = [
                    "uri", "name", 
                    "id", "<id>", "elementId", "<elementId>", 
                    "inferred_category", "source_category", # 兼容您之前的分类字段
                    "source_chunk", "source_doc",
                    "created_from", "first_source", "promoted", 'inferred_category', 'is_promoted', 'name_optimized', 
                  'promotion_date', 'promotion_degree', 'source_category', 
                  'source_chunk', 'source_doc', 'name', 'entities_before_merge', 
                  'merged_from_history', 'labels_before_merge', 'is_merged', 
                  'original_name_before_opt','crystalsystem_source','shape_source'# 保险起见，把之前写过的系统字段全加上
                ]
                
                # 批量剔除
                for key in exclude_keys:
                    props.pop(key, None)
                    
                # 【可选增强】：如果您之前还为具体属性加上了后缀（如 shape_source_doc），可以用下面这行把带后缀的溯源也删掉
                keys_to_delete = [k for k in props.keys() if k.endswith("_source_doc") or k.endswith("_source_chunk")]
                for k in keys_to_delete:
                    props.pop(k, None)
                
                # 2. 【核心拦截】：如果剔除黑名单后字典为空，说明只有溯源信息，没有实质的物理/化学属性，直接跳过
                if not props:
                    continue
                    
                # 3. 只有包含实质性额外属性时，才遍历格式化
                prop_lines = [f"    - {k}: {v}" for k, v in props.items()]
                prop_blocks.append(f"  [{e_name} 的内部属性]:\n" + "\n".join(prop_lines))
            
            if prop_blocks:
                # 【方案 A 核心】：仅针对海量且容易干扰视线的属性信息，利用 LLM 进行物理约束清洗
                print(f"  [LLM 智能筛选] 正在基于物理约束清洗实体属性噪点...")
                filtered_props = self._filter_attributes_via_llm(question, prop_blocks)
                properties_context = "[实体内部属性 (Entity Attributes)]:\n" + "\n".join(filtered_props) + "\n\n"
            # ================= 通道二：图谱拓扑寻路 (前置助手函数) =================
            def execute_path_cypher(cypher_query, path_limit=None, **kwargs):
                paths = set()
                results = session.run(cypher_query, **kwargs)
                for record in results:
                    nodes = record["path_nodes"]
                    rels = record["path_rels"]
                    path_str = f"({nodes[0]})"
                    for i in range(len(rels)):
                        path_str += f" -[{rels[i]}]-> ({nodes[i+1]})"
                    paths.add(path_str)
                
                # 【新增：检索超限时的 LLM 智能裁切】
                if path_limit and len(paths) > path_limit:
                    paths = set(list(paths)[:path_limit])
                return paths
            # ================= 策略 A：单实体检索 =================
            if len(entities) == 1:
                entity = entities[0]
                
                # 【特判】：是否为纯粹的分类溯源查询
                is_pure_classification = len(target_relations) > 0 and set(target_relations).issubset({"subclassOf", "isAKindOf"})
                
                if is_pure_classification:
                    taxonomy_cypher = """
                    MATCH (n)
                    WHERE n.name = $entity AND (n:Term OR n:Other OR n:Classification)
                    MATCH p = (n)-[:isAKindOf|subclassOf*1..4]->(m:Classification)
                    // 【核心优化】：限定 m 必须是最尽头的节点（不能再有向上的分类关系）
                    WHERE NOT (m)-[:isAKindOf|subclassOf]->(:Classification)
                      AND ALL(r IN relationships(p) WHERE (startNode(r):Classification OR endNode(r):Classification) OR r.in_subgraph_1_3 = true)   /* === [ADD] Term/Other 间关系需 1/3 子图 === */
                    RETURN [x IN nodes(p) | x.name] AS path_nodes, [x IN relationships(p) | type(x)] AS path_rels
                    LIMIT 50
                    """
                    core_triplets = execute_path_cypher(taxonomy_cypher, entity=entity)

                if len(target_relations) > 0:
                    rel_pattern = "|".join(target_relations)
                    core_cypher = f"""
                    MATCH p = (n)-[r:{rel_pattern}]-(m)
                    WHERE n.name = $entity AND (n:Term OR n:Other OR n:Classification) AND (m:Term OR m:Other OR m:Classification)
                      AND ((n:Classification OR m:Classification) OR r.in_subgraph_1_3 = true)   /* === [ADD] Term/Other 间关系需 1/3 子图 === */
                    RETURN [x IN nodes(p) | x.name] AS path_nodes, [x IN relationships(p) | type(x)] AS path_rels
                    LIMIT 50
                    """
                    core_triplets.update(execute_path_cypher(core_cypher, entity=entity))
                
                broad_cypher = """
                MATCH p = (n)-[*1..2]-(m)
                WHERE n.name = $entity AND (n:Term OR n:Other OR n:Classification) AND (m:Term OR m:Other OR m:Classification)
                AND ALL(r IN relationships(p) WHERE (startNode(r):Classification OR endNode(r):Classification) OR r.in_subgraph_1_3 = true)   /* === [ADD] Term/Other 间关系需 1/3 子图 === */
                AND (
                    length(p) = 2
                    OR size([(m)--(next) WHERE next <> n AND (next:Term OR next:Other OR next:Classification) | 1]) = 0
                )
                RETURN [x IN nodes(p) | x.name] AS path_nodes, [x IN relationships(p) | type(x)] AS path_rels
                LIMIT 100
                """
                raw_candidates = execute_path_cypher(broad_cypher, entity=entity)
                candidate_triplets.update(raw_candidates - core_triplets)
            # ================= 策略 B：多实体协同检索 (>= 2个实体) =================
            else:
                # 【机制 1：多实体分类溯源特判】
                has_classification_relation = any(rel in ["subclassOf", "isAKindOf"] for rel in target_relations)
                
                if has_classification_relation:
                    taxonomy_cypher = """
                    MATCH (n)
                    WHERE n.name IN $entities AND (n:Term OR n:Other OR n:Classification)
                    MATCH p = (n)-[:isAKindOf]->(:Classification)-[:subclassOf*1..4]->(m:Classification)
                    WHERE ALL(r IN relationships(p) WHERE (startNode(r):Classification OR endNode(r):Classification) OR r.in_subgraph_1_3 = true)   /* === [ADD] Term/Other 间关系需 1/3 子图 === */
                    RETURN [x IN nodes(p) | x.name] AS path_nodes, [x IN relationships(p) | type(x)] AS path_rels
                    LIMIT 50
                    """
                    core_triplets.update(execute_path_cypher(taxonomy_cypher, entities=entities))

                # 【机制 2：多实体全局最短路径，构建"核心骨架 (Core Paths)"】
                # 说明: id(n) < id(m) 是无向图配对的神仙操作，能确保实体列表中任意两个实体都会且只会被计算一次最短路径。
                path_cypher = """
                MATCH (n) WHERE n.name IN $entities AND (n:Term OR n:Other OR n:Classification)
                WITH collect(n) AS nodes
                UNWIND nodes AS n
                UNWIND nodes AS m
                WITH n, m WHERE id(n) < id(m)
                MATCH p = shortestPath((n)-[*1..5]-(m))
                WHERE ALL(x IN nodes(p) WHERE x:Term OR x:Other OR x:Classification)
                  AND ALL(r IN relationships(p) WHERE (startNode(r):Classification OR endNode(r):Classification) OR r.in_subgraph_1_3 = true)   /* === [ADD] Term/Other 间关系需 1/3 子图 === */
                RETURN [x IN nodes(p) | x.name] AS path_nodes, [x IN relationships(p) | type(x)] AS path_rels
                """
                # AND ALL(r IN relationships(p) WHERE size($valid_rels) = 0 OR type(r) IN $valid_rels)
                core_path_nodes = set(entities)
                temp_shortest_paths = set()
            
                path_results = session.run(path_cypher, entities=entities, valid_rels=target_relations)
                for record in path_results:
                    nodes = record["path_nodes"]
                    rels = record["path_rels"]
                    core_path_nodes.update(nodes)
                    path_str = f"({nodes[0]})"
                    for i in range(len(rels)):
                        path_str += f" -[{rels[i]}]-> ({nodes[i+1]})"
                    temp_shortest_paths.add(path_str)
                
                # 【机制特供：最短路径同样享受 LLM 筛选】
                if len(temp_shortest_paths) > 15:
                    temp_shortest_paths = set(list(temp_shortest_paths)[:15])
                core_triplets.update(temp_shortest_paths)
                
                # 【机制 3【外围辐射补充】：核心骨架的 1 跳辐射扩展 (Candidate Paths)】
                # 基于上述寻路收集到的所有节点（包含原始实体 + 最短路径途经的中间实体），向外辐射一跳
                if core_path_nodes:
                    # 方案 1：优先根据意图定向拉取 (限额高达 1500，彻底防止截断)
                    if target_relations:
                        rel_pattern = "|".join(target_relations)
                        rad_cypher_targeted = f"""
                        MATCH (n) WHERE n.name IN $core_nodes
                        MATCH p = (n)-[r:{rel_pattern}]-(neighbor)
                        WHERE (neighbor:Term OR neighbor:Other OR neighbor:Classification)
                          AND ((n:Classification OR neighbor:Classification) OR r.in_subgraph_1_3 = true)   /* === [ADD] Term/Other 间关系需 1/3 子图 === */
                        RETURN [x IN nodes(p) | x.name] AS path_nodes, [x IN relationships(p) | type(x)] AS path_rels
                        """
                        targeted_rad = execute_path_cypher(rad_cypher_targeted, path_limit=1500, core_nodes=list(core_path_nodes))
                        candidate_triplets.update(targeted_rad - core_triplets)
                    
                    # 方案 2：无差别兜底防漏 (限额 1000)
                    rad_cypher_broad = """
                    MATCH (n) WHERE n.name IN $core_nodes
                    MATCH p = (n)-[*1]-(neighbor)
                    WHERE (neighbor:Term OR neighbor:Other OR neighbor:Classification)
                      AND ALL(r IN relationships(p) WHERE (startNode(r):Classification OR endNode(r):Classification) OR r.in_subgraph_1_3 = true)   /* === [ADD] Term/Other 间关系需 1/3 子图 === */
                    RETURN [x IN nodes(p) | x.name] AS path_nodes, [x IN relationships(p) | type(x)] AS path_rels
                    """
                    broad_rad = execute_path_cypher(rad_cypher_broad, path_limit=1000, core_nodes=list(core_path_nodes))
                    candidate_triplets.update(broad_rad - core_triplets)
        # ================= 阶段二：带主干免疫的语义剪枝 (Core-Immune Pruning) =================
        core_list = list(core_triplets)
        candidate_list = list(candidate_triplets)
        
        graph_context_block = ""
        
        if not core_list and not candidate_list:
            graph_context_block = "  - 未检索到相关的图谱拓扑路径 (No graph paths retrieved)."
        else:
            if len(core_list) + len(candidate_list) <= max_triplets_limit:
                final_paths = core_list + candidate_list
            else:
                remaining_slots = max_triplets_limit - len(core_list)
                if remaining_slots <= 0:
                    final_paths = core_list
                else:
                    question_embedding = self.embedding_model.encode([question])
                    triplet_embeddings = self.embedding_model.encode(candidate_list)
                    similarities = cosine_similarity(question_embedding, triplet_embeddings)[0]
                    top_k_indices = np.argsort(similarities)[-remaining_slots:][::-1]
                    pruned_candidates = [candidate_list[i] for i in top_k_indices]
                    final_paths = core_list + pruned_candidates
            
            graph_context_block = "[图谱拓扑路径 (Graph Topology Paths)]:\n  " + "\n  ".join(final_paths)

        # ================= 完美合体：属性 + 路径 =================
        return properties_context + graph_context_block

    def generate_answer(self, question, context):   
        """
        步骤 5：最终答案生成。
        全能融合版 Prompt：既能处理属性和普通多跳，又完美继承了材料学专属的分类层级寻路逻辑。
        """
        prompt = f"""
你是一个严谨的材料科学问答系统。你的任务是严格根据提供的[图谱知识事实]来精确回答用户的[问题]。

【全局约束条件】：
1. 事实至上：你的答案必须 100% 基于[图谱知识事实]中的文本推导得出，严禁使用任何先验外部知识。
2. 极简输出：只输出最终的核心答案（例如具体的材料名、属性值、分类名、测试设备等），不要输出完整的句子，不要包含任何解释或多余的标点。
3. 未知处理：如果在事实中完全找不到推导依据，请输出 "Unknown"。

【多维度推理逻辑说明】：
- 🔍 [属性查询]：如果问题询问特定的物理/化学特性（如熔点、形状），请直接从实体内部属性块中提取数值。
- 🔗 [拓扑寻路]：如果问题涉及实体间的直接交互或多跳关联（例如A组成B，B用于C），请依据 `(节点)-[关系]->(节点)` 路径一步步推导。
- 🌳 [分类层级解析规则 (Taxonomy Logic)]：
   1. 实体(Term) 通过 `isAKindOf` 链接到它的【直接分类】。
   2. 【直接分类】通过 `subclassOf` 进一步链接到更高级、更广义的【父级分类】。
   3. 如果问题要求寻找“更广义(broader)”、“宏观分类(macro classification)”或“包含直接分类”的类别，你应该沿着 `subclassOf` 路径向上寻找更高一层的节点。
   4. 如果事实显示 A -> B -> C，且问题问的是“包含B的更广义分类”，你应该回答 C。

[图谱知识事实]:
{context}

[用户问题]:
{question}
"""

        completion = self.llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个基于图谱事实的严谨学术助手，绝不编造信息，遵循层级推导规则，且只输出最精简的核心答案。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01  # 保持极低的温度，确保推理的确定性和事实的绝对忠诚
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
    def run_pipeline(self, question):
        """运行完整的问答流水线，并返回中间过程数据"""
        intent = self.extract_intent_via_llm(question)
        print(f"意图分析结果: {intent}")
        mentions = intent["entities"]
        target_relations = intent["relations"]
        
        linked_entities = self.link_entities(mentions)
        
        # 检索时传入 target_relations
        context = self.retrieve_subgraph(linked_entities, target_relations, question)
        print("检索到的图谱上下文:\n", context)
        answer = self.generate_answer(question, context)
        
        return {
            "raw_mentions": mentions,
            "target_relations": target_relations,  # 记录大模型猜中的关系
            "linked_entities": linked_entities,
            "retrieved_context": context,
            "generated_answer": answer
        }

# ================= 自动化评测逻辑 =================

def evaluate_dataset(dataset_path, output_path):
    """
    运行整个数据集的评测并计算准确率。
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    pipeline = HybridKBQAPipeline()
    pipeline.build_graph_vocabulary()
    
    results_log = []
    correct_count = 0

    print(f"\n[评测日志] 开始执行 SG-RAG 评测，总计 {len(qa_data)} 道题目...\n")

    for idx, item in enumerate(qa_data):
        qid = item["id"]
        # 默认使用英文问题和标准答案进行测试
        question = item["question_en"]
        gold_answer = item["gold_answer_en"]
        
        print(f"[{qid}] 正在处理问题: {question}")
        
        result = pipeline.run_pipeline(question)
        llm_answer = result["generated_answer"]
        
        # 使用 Qwen LLM 进行语义判定
# 注意：你需要将你的测试问题 (question) 传入，这有助于大模型理解上下文
        is_correct = pipeline.judge_with_qwen(
        question=question,     # 你的原问题变量名
        gold_answer=gold_answer, 
        llm_answer=llm_answer, 
        model_name="qwen-max"          # 替换为你实际使用的 Qwen 模型名称 (如 qwen-max, qwen-turbo 或本地路径)
        )
        if is_correct:
           correct_count += 1
            
        print(f"   -> LLM 提取词汇 : {result['raw_mentions']}")
        print(f"   -> 链接图谱实体 : {result['linked_entities']}")
        print(f"   -> 模型预测结果 : {llm_answer} (标准答案: {gold_answer})")
        print(f"   -> 评测状态     : {'[通过 ✅]' if is_correct else '[失败 ❌]'}\n")
        
        # 保存详细的推理过程日志
        item.update({
            "pipeline_mentions": result["raw_mentions"],
            "pipeline_linked": result["linked_entities"],
            "retrieved_context": result["retrieved_context"],
            "llm_prediction": llm_answer,
            "is_correct": is_correct
        })
        results_log.append(item)


        # ================= 🌟 新增：每 50 题自动保存一次 =================
        if (idx + 1) % 50 == 0:
            current_acc = correct_count / (idx + 1)
            print(f"\n💾 [Checkpoint] 已处理 {idx + 1} 题，当前准确率: {current_acc*100:.2f}%。正在写入本地缓存...")
            # 及时覆写保存到文件，防止中途崩溃
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "experiment_summary": {
                        "total_questions_in_dataset": len(qa_data),
                        "processed_so_far": idx + 1,
                        "current_accuracy": f"{current_acc*100:.2f}%"
                    },
                    "detailed_results": results_log
                }, f, ensure_ascii=False, indent=4)
            print("✅ 进度保存成功！继续执行...\n")

    pipeline.close()

    accuracy = correct_count / len(qa_data) * 100
    print(f"[实验总结] 评测结束！最终准确率: {accuracy:.2f}% ({correct_count}/{len(qa_data)})")

    # ================= [ADD] 按类型和来源统计正确率 =================
    type_stats = {}
    source_stats = {}
    for r in results_log:
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

    # ================= 🌟 修改：输出包含最终准确率的结构 =================
    final_output = {
        "experiment_summary": {
            "total_questions": len(qa_data),
            "correct_count": correct_count,
            "final_accuracy": f"{accuracy:.2f}%",
            # === [ADD] 各类型问题正确率 ===
            "accuracy_by_type": {qt: f"{st['correct']/st['total']*100:.2f}%" for qt, st in type_stats.items()},
            # === [ADD] 不同来源正确率 ===
            "accuracy_by_source": {src: f"{ss['correct']/ss['total']*100:.2f}%" for src, ss in source_stats.items()},
        },
        "detailed_results": results_log
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    # ====================================================================
    print(f"[系统日志] 详细的实验结果记录已保存至 {output_path}")

if __name__ == "__main__":
    # 填入你的测试集文件路径
    evaluate_dataset("/root/files_lpz/ra/实验/literature/qa_dataset/qa_dataset_merged.json", "/root/files_lpz/ra/实验/literature/exp_code/kg_scale/13scale_kgqa_evaluation_results_split.json")