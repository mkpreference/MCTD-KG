

from py2neo import Graph
import pandas as pd
import os
def clear_folder(folder_path):
    # 检查路径是否存在且是一个文件夹
    if os.path.isdir(folder_path):
        # 遍历文件夹中的所有文件和文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # 如果是文件，则删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件
                # 注意：这里不处理子文件夹，如果你也想删除子文件夹，可以使用shutil.rmtree
            except Exception as e:
                print(f"无法删除 {file_path}。原因: {e}")
    else:
        print(f"{folder_path} 不是一个有效的文件夹路径。")
def build_graph_1(enitys,enity_types,relations,relation_types):
    filePath = 'D:/app/neo4j-community-4.4.30/import'
    files = os.listdir(filePath)
    print(files)
    # graph = Graph("http://localhost:7474", username="neo4j", password="LPZ12256zpl")
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "LPZ12256zpl"))



    # 创建Person节点
    for file in files:
        if "_" not in file:
            index = file[:-4]
            print(index)
            create_person_query = f"LOAD CSV WITH HEADERS FROM \"file:///{file}\" AS row MERGE (p:{index} {{name: row.{index}}});"

            print(create_person_query)
            graph.run(create_person_query)

    for file in files:
        if "_"  in file:
            # 创建关系
            index = file[:-4]
            n = index.find('_')
            enity_type_1 = index[0:n]
            enity_type_2 = index[n+1:]
            create_relationship_query = f"LOAD CSV WITH HEADERS FROM \"file:///{file}\" AS row MATCH (start:{enity_type_1} {{name: row.start_name}}), (end:{enity_type_2} {{name: row.end_name}}) CALL apoc.create.relationship(start, row.type, {{}}, end) YIELD rel RETURN rel;"
            print(create_relationship_query)
            graph.run(create_relationship_query)
def build_graph(enitys,enity_types,relations,relation_types):
    filePath = 'D:/app/neo4j-community-4.4.30/import'
    files = os.listdir(filePath)
    print(files)
    # graph = Graph("http://localhost:7474", username="neo4j", password="LPZ12256zpl")
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "LPZ12256zpl"))

    # 创建节点
    df = pd.read_excel(filePath + "/entitys.xlsx")
    for i in range(df.shape[0]):
        data = df.at[i, "entity"]
        data, enity_type = replace(data)
        create_person_query = f"MERGE (a:`{enity_type}` {{name:\"{data}\"}});"

        print(create_person_query)
        graph.run(create_person_query)
    #添加属性
    # MERGE(p: Person
    # {name: 'John'})
    # SET
    # p.age = 30, p.email = 'john@example.com';
    df = pd.read_excel(filePath + "/attributes.xlsx")
    for i in range(df.shape[0]):
        data = df.at[i, "entity"]
        attribute = df.at[i, "attribute"]
        value = df.at[i, "value"]
        data, enity_type = replace(data)
        create_person_query = f"MERGE (p:`{enity_type}` {{name: \"{data}\"}}) SET p.`{attribute}` = \"{value}\";"
        print(create_person_query)
        graph.run(create_person_query)
    df = pd.read_excel(filePath + "/output.xlsx")
    for i in range(df.shape[0]):
        enity1 = df.at[i, "head"]
        enity2 = df.at[i, "tail"]
        enity1, enity1_type = replace(enity1)
        enity2, enity2_type = replace(enity2)
        relation = df.at[i, "relation"]
        create_relationship_query = f"match(n:`{enity1_type}`{{name:\"{enity1}\"}}),(m:`{enity2_type}`{{name:\"{enity2}\"}}) create (n)-[r:{relation}]->(m);"
        print(create_relationship_query)
        graph.run(create_relationship_query)
def replace(data):
    enity_type = data.replace(" ","_")
    enity_type = enity_type.replace(",", "_")
    return data,enity_type
if __name__ == "__main__":
    build_graph(1,1,1,1)
    folder_to_clear = 'D:/app/neo4j-community-1_4.1_4.30/import' # 替换为你的文件夹路径
    clear_folder(folder_to_clear)