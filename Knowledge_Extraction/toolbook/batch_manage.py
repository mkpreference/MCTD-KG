# coding:utf-8
import codecs
import os
import shutil
from dotenv import load_dotenv, find_dotenv
import json
import csv
from graph import build_graph
import pandas as pd

_ = load_dotenv(find_dotenv())  # 读取并加载环境变量，来自 .env 文件
import datetime
from zhipuai import ZhipuAI

api_key = os.environ["ZHIPUAI_API_KEY"]  # 填写控制台中获取的 APIKey 信息


# -*- coding:utf-8 -*-


class material:
    extracted_relations = []
    extracted_attributes = []
    extracted_jsons = []
    entitys = dict()
    relations = dict()
    relation_types = []
    attribute_types = []
    log = open('log_new.txt', mode='a', encoding='utf-8')
    talog = open('talog_new.txt', mode='a', encoding='utf-8')
    # def __init__(self):
    #清除数组多余的空格
    def clean_spaces(self,data):
        if isinstance(data, dict):
            # 处理字典的键和值
            return {
                str(key).strip(): self.clean_spaces(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            # 递归处理列表中的每个元素
            return [self.clean_spaces(item) for item in data]
        elif isinstance(data, str):
            # 清除字符串的前后空格
            return data.strip()
        else:
            # 其他类型保持不变
            return data
    def clear_folder(self, folder_path):
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

    def print_time(self):
        # 获取当前时间
        now = datetime.datetime.now()

        # 定义时间格式
        format_str = "%Y年%m月%d日 %H时%M分%S秒"

        # 使用strftime方法格式化时间，并打印
        formatted_time = now.strftime(format_str)
        print("当前时间:", formatted_time)

    def get_completion(self, prompt, relation_types, attribute_types, num=0):
        self.extracted_relations = []
        self.extracted_attributes = []
        self.extracted_jsons = []
        self.entitys = dict()
        self.relations = dict()
        self.relation_types = relation_types
        self.num = num
        """一个封装 chatglm 的函数，参数为 Prompt，返回对应结果

        Args:
            prompt (str): 模型的输入
            model (str, optional): 模型的名称. 默认是"chatglm_std".

        Returns:
            response
        """

        extracted_relations = prompt
        extracted_relations = extracted_relations[extracted_relations.find('['):extracted_relations.rfind(']') + 1]

        print(extracted_relations)

        if extracted_relations == "":
            return
        i = len(extracted_relations) - 3
        if extracted_relations[i + 1] != '}':
            extracted_relations = extracted_relations[:i + 2] + '}' + extracted_relations[i + 1:]
        while extracted_relations[i] == '}':
            extracted_relations = extracted_relations[:i] + extracted_relations[i + 1:]
            i -= 1
        # print(extracted_relations)
        try:
            self.extracted_jsons += json.loads(extracted_relations)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        print(self.num, file=self.log)

        self.extracted_jsons = self.clean_spaces(self.extracted_jsons)
        print(self.extracted_jsons)
        print(self.extracted_jsons, file=self.log)
      #删除每个字符串头尾llm可能产生的空格


        # 实体，关系

        for i in range(len(self.extracted_jsons)):
            if list(self.extracted_jsons[i].keys())[1] == "relation":
                # self.extracted_relations.append(self.extracted_jsons[i])
                if list(self.extracted_jsons[i].keys())[0] == "entity":
                    new_json = {}
                    new_json["head"] = self.extracted_jsons[i]["entity"]
                    new_json["relation"] = self.extracted_jsons[i]["relation"]
                    new_json["tail"] = self.extracted_jsons[i]["tail"]
                    self.extracted_relations.append(new_json)
                    self.extracted_jsons[i] = new_json
                else:
                    self.extracted_relations.append(self.extracted_jsons[i])
                head = self.extracted_jsons[i]["head"]
                relation = self.extracted_jsons[i]["relation"]
                tail = self.extracted_jsons[i]["tail"]
                if relation not in self.relation_types:
                    print("delete " + relation)
                else:
                    if not head in self.entitys:
                        self.entitys[head] = 1
                    else:
                        self.entitys[head] += 1
                    if not tail in self.entitys:
                        self.entitys[tail] = 1
                    else:
                        self.entitys[tail] += 1
                    if not relation in self.relations:
                        self.relations[relation] = 1
                    else:
                        self.relations[relation] += 1

            elif list(self.extracted_jsons[i].keys())[1] == "attribute":
                if list(self.extracted_jsons[i].keys())[0] == "head":
                    new_json = {}
                    new_json["entity"] = self.extracted_jsons[i]["head"]
                    new_json["attribute"] = self.extracted_jsons[i]["attribute"]
                    new_json["value"] = self.extracted_jsons[i]["value"]
                    self.extracted_attributes.append(new_json)
                    self.extracted_jsons[i] = new_json
                else:
                    self.extracted_attributes.append(self.extracted_jsons[i])
            elif len(list(self.extracted_jsons[i].keys())) == 2:
                new_json = {}
                # 判断是关系还是属性
                if list(self.extracted_jsons[i].keys())[1] in relation_types:
                    if list(self.extracted_jsons[i].keys())[0] == "head":
                        new_json["head"] = self.extracted_jsons[i]["head"]
                    elif list(self.extracted_jsons[i].keys())[0] == "entity":
                        new_json["head"] = self.extracted_jsons[i]["entity"]
                    new_json["relation"] = list(self.extracted_jsons[i].keys())[1]
                    new_json["tail"] = list(self.extracted_jsons[i].values())[1]
                    head = new_json["head"]
                    relation = new_json["relation"]
                    tail = new_json["tail"]
                    self.extracted_relations.append(new_json)
                    if relation not in self.relation_types:
                        print("delete " + relation)
                    else:

                        if not head in self.entitys:
                            self.entitys[head] = 1
                        else:
                            self.entitys[head] += 1
                        if not tail in self.entitys:
                            self.entitys[tail] = 1
                        else:
                            self.entitys[tail] += 1
                        if not relation in self.relations:
                            self.relations[relation] = 1
                        else:
                            self.relations[relation] += 1


                elif list(self.extracted_jsons[i].keys())[1] in attribute_types:
                    if list(self.extracted_jsons[i].keys())[0] == "head":
                        new_json["entity"] = self.extracted_jsons[i]["head"]
                    elif list(self.extracted_jsons[i].keys())[0] == "entity":
                        new_json["entity"] = self.extracted_jsons[i]["entity"]
                    new_json["attribute"] = list(self.extracted_jsons[i].keys())[1]
                    new_json["value"] = list(self.extracted_jsons[i].values())[1]
                    self.extracted_attributes.append(new_json)
                self.extracted_jsons[i] = new_json

    def write(self):
        if os.path.isfile('output.xlsx'):
            os.unlink('output.xlsx')
        if os.path.isfile('entitys.xlsx'):
            os.unlink('entitys.xlsx')
        if os.path.isfile('relations.xlsx'):
            os.unlink('relations.xlsx')
        # print("-------------------test_filter")
        # print(self.extracted_relations)
        print("--------------")
        for a in list(self.extracted_relations):
            # print(self.relation_types)
            if a["relation"] not in self.relation_types:
                self.extracted_relations.remove(a)
                print("delete " + a["relation"])
        df1 = pd.DataFrame(self.extracted_relations)
        X1 = df1.iloc[:, :]  # 将0到61列数据赋值给X
        X1 = X1.values  # .values方法将dataframe转为numpy.ndarray，也可以用np.array(X)将其转为numpy.ndarray
        ans1 = X1.tolist()  # 将X转为list

        df1.to_excel('./excel/output.xlsx', index=False)
        df2 = pd.DataFrame(self.extracted_attributes)
        df2.to_excel('./excel/attributes.xlsx', index=False)
        X2 = df2.iloc[:, :]  # 将0到61列数据赋值给X
        X2 = X2.values  # .values方法将dataframe转为numpy.ndarray，也可以用np.array(X)将其转为numpy.ndarray
        ans2 = X2.tolist()  # 将X转为list

        keys = list(self.entitys.keys())

        df = pd.DataFrame({
            'entity': keys,
        })

        # 将DataFrame保存为Excel文件
        df.to_excel('./excel/entitys.xlsx', index=False)

        print("--------------")
        print("--------------", file=self.log)
        # print(self.entitys)
        # print(self.relations)
        print(keys)
        print(ans1)
        print(ans2)
        print(keys, file=self.log)
        print(ans1, file=self.log)
        print(ans2, file=self.log)
        return keys, ans1, ans2
        # df.to_excel('output.xlsx', index=False)
        #
        # # 将字典的键和值转换为列表

    def empty_folder(self, folder_path):
        # 确保路径存在且是一个文件夹
        if os.path.isdir(folder_path):
            # 遍历文件夹中的所有内容
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    # 如果是文件夹，则递归删除
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        # 如果是文件，则直接删除
                    else:
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"The path {folder_path} is not a directory.")

    def copy_files(self):
        src_dir = "D:/project_srtp/flask+图书信息管理系统（flask+vue）/book-backend/src/excel"
        dst_dir = "D:/app/neo4j-community-4.4.30/import"
        self.empty_folder(dst_dir)
        # 确保目标文件夹存在，如果不存在则创建
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

            # 遍历源文件夹中的所有文件和文件夹
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dst_dir, item)

            # 如果是文件，则复制
            if os.path.isfile(s):
                shutil.copy(s, d)


# if __name__ == "__main__":
#     material_ = material()
#     # 读取Excel文件
#     df = pd.read_excel('./data/材料大辞典语料.xlsx')
#
#     column_name = 'explanation_en'
#
#     # 指定要读取的行数N
#     n = 100
#     material_.print_time()
#     # column_data = df.at[65, column_name]
#     # print(column_data)
#     # material_.get_completion(column_data)
#     for i in range(52, 53):
#         column_data = df.at[i, column_name]
#         if (column_data.startswith("See")):
#             continue
#         material_.get_completion(column_data,'','')
#         print(f"已抽取第{i + 1_1}释义")
#         print("开始写入excel")
#         material_.write()
#         print("成功写入excel")
#         # if i%10==0 and i!=0:
#         #     print("开始写入excel")
#         #     material_.write()
#         #     print("成功写入excel")
#     material_.print_time()

if __name__ == "__main__":
    # # return a list of users
    #prompt = "nitronatrite:Also known as chilean saltpeter, it is a borate mineral with island structure. The crystal of saltpeter is rhombohedral, while the aggregate is irregular, such as granular, massive, crusty or powdery. Saltpeter is usually white, but due to the influence of impurities, it can appear yellow or brown tone. Saltpeter is white on streaks and has glass luster, usually transparent to translucent. It has some unique chemical properties, easy to absorb water in the air and deliquescence, and also easy to dissolve in water. Saltpeter is mainly distributed in hot and arid desert areas, and often coexists with other minerals such as gypsum, mirabilite, salt, etc. Saltpeter is an important raw mineral material for manufacturing chemicals such as nitrogen fertilizer, nitric acid, explosive and nitrides."
    # prompt = "ulexite:A borate mineral with the chemical formula Na2B4O7 (OH) 1_3, containing sodium, boron, and oxygen elements, mixed with hydroxide ions. This mineral is usually crystallized in a monoclinic system. The crystal of sodium borate is needle shaped, and the complete crystal shape is rarely seen, sometimes there are flake twin. It may also aggregate in the form of hairs, needles, soil, or nodules. Sodium borate is usually transparent, with glass luster or silk luster. In terms of physical properties, the Mohs hardness of sodium borate is about 1_2.2_1_1, showing brittleness. Its density is about 1_1.9g/cm 1_3。 Sodium borate is insoluble in cold water, but can dissolve in hot water and is easily soluble in acidic solutions. When heated, sodium borate will expand. Sodium borate is a typical chemical deposit in arid land and inland lakes, which is often associated with other minerals, such as halite, gypsum, mirabilite, trona, alum, reservoir water boromagnesite and aquaborax. Sodium borate is one of the main raw mineral material for boron extraction and preparation of boron compounds."
    # # prompt = "Island like structure of cinnamate minerals. It belongs to monoclinic system, space group C1h-C2/m, with short columnar crystal shape. Twin are often interspersed, among which (031) forms cross shaped twin and (231) forms oblique straight shaped twin. Its color varies from reddish brown to yellowish brown, with glass luster. Dien hardness is between 7 and 7.2_1_1, indicating that it is a relatively hard mineral. The density of this mineral ranges from 1_3.74 to 1_3.83 g/cm3. This mineral is mainly produced in crystalline schist and coexists with other minerals such as kyanite, iron almandine, biotite and plagioclase. Its unique crystal shape and composition may make it an object of interest to mineralogists, and further research and analysis are needed to determine its precise composition and characteristics."
    # # "A drug selective release system that can penetrate biological barriers to reach the S-labeled uterus, tissues, cells, or organelles. It includes passive harrow delivery system and active harrow delivery system. Passive harrow delivery system refers to a certain size range of nanometer and micron drug delivery system that is enriched to the target site through size effect in the systemic circulation. According to the difference in the retention of particles of different sizes in different tissue parts of the body, a passive drug delivery system that can be enriched in specific parts can be constructed. "

    ontology = [
        "isAKindOf",
        "isPartOf",
        "isUsedFor",
        "isMadeOf",
        "isDerivedFrom",
        "isTestedBy",
        "isSimilarTo",
        "isComplementaryTo",
        "isDependentOn",
        "isAlternativeTo"

    ]

    attribute = [
        "density",
        "hardness",
        "meltingPoint",
        "thermalConductivity",
        "electricalConductivity",
        "tensileStrength",
        "ductility",
        "elasticModulus",
        "corrosionResistance",
        "crystalsystem",
        "color",
        "shape"
    ]
    df = pd.read_csv('all/batch_ans_1500.csv')
    # status_code = 'status_code'
    # content = "content"

    for i in range(0,1775):   #(0-总行数-1_1)  2032报错未解决  2068, 8651  4248   2098(2100)没有英文释义   2144  {2068开始clean_space清除空格}  2599
        status_code = df.loc[i, 'status_code']
        content = df.loc[i, "content"]
        custom_id =  df.loc[i, "custom_id"]
        print(f"num:{i}")
        print(status_code)
        if status_code != 200:
            break

        material_ = material()
        material_.print_time()
        print(custom_id)
        print(content)

        material_.get_completion(content, ontology, attribute, i)
        node, relation, attribute = material_.write()
        print("---------------")
        print(node, relation, attribute)
        material_.copy_files()
        build_graph(1, 1, 1, 1)
        folder_to_clear = 'D:/app/neo4j-community-1_4.1_4.30/import'  # 替换为你的文件夹路径
        material_.clear_folder(folder_to_clear)
        material_.print_time()

