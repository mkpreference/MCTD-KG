import numpy as np
from neo4j import GraphDatabase

def analyze_degree_distribution(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # 仅统计当前仍为 Other 且尚未被 LLM 拒绝的节点
    query = """
    MATCH (n:Other)
    WHERE n.term_rejected IS NULL
    WITH n, size((n)--()) AS degree
    RETURN degree, count(n) AS node_count
    ORDER BY degree
    """
    
    degrees = []
    with driver.session() as session:
        results = session.run(query)
        print("📊 节点度频数分布:")
        for record in results:
            degree = record["degree"]
            count = record["node_count"]
            print(f"  - 关系数为 {degree} 的节点有 {count} 个")
            # 展开为一维数组以便于计算分位数
            degrees.extend([degree] * count)
            
    driver.close()
    
    if not degrees:
        print("未找到符合条件的 Other 节点。")
        return

    degrees = np.array(degrees)
    print("\n📈 统计学特征分析:")
    print(f"  - 总节点数: {len(degrees)}")
    print(f"  - 平均关系数: {np.mean(degrees):.2f}")
    print(f"  - 最大关系数: {np.max(degrees)}")
    print(f"  - 75% 分位数 (前 25% 的节点度 >=): {np.percentile(degrees, 75)}")
    print(f"  - 90% 分位数 (前 10% 的节点度 >=): {np.percentile(degrees, 90)}")
    print(f"  - 95% 分位数 (前 5% 的节点度 >=): {np.percentile(degrees, 95)}")
    
    print("\n💡 阈值设定建议:")
    print("  1. 严格模式 (取 95% 分位数): 仅让处于头部的极少数高频节点触发大模型晋升验证，节省 Token，保证核心术语绝对权威。")
    print("  2. 平衡模式 (取 90% 分位数): 适合当前图谱文献融合量适中的阶段。")
    print("  3. 长尾截断法 (Elbow Method): 观察频数分布打印，找到数量发生断崖式下跌的那个 degree 作为阈值。")

if __name__ == "__main__":
    analyze_degree_distribution("bolt://localhost:7687", "neo4j", "LPZ12256zpl")