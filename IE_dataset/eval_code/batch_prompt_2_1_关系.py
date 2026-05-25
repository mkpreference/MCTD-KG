import csv
import json
import pickle
def messages_builder_example(content1,content2):
    messages = [{"role": "system", "content": "You are an expert agent who specializes in analyzing a large dictionary of material, and your task is to extract Entities, relationships between entities from given interpretations of material terms.The relationships must be one of the types in 'relation_types' in the user list provided." +
                                    "Please extract the entities and relationships in the user tips and generate the output in JSON format according to the following material terms interpretation.The output should be a json list, without any extra line breaks, Spaces, or useless prompt language, " +
                                    " --> Beginning of the  case of json: the extraction of entity relationship" +
                                    "If you are extracting relationships between entities, the json object must contain the following keywords;\"head\", \"relation\", \"tail\"" +
                                    "The \"relation\" keywords must be one of the types in 'relation_types' in the user list provided." +
                                    " The \"head\" represents the primary entity or subject in a relationship." +
                                    'The "relation" describes the link or connection between the "head" and "tail" entities ,and The \"relation\" keywords must be one of the types in "relation_types" in the user list provided.                  ' +
                                    'The "tail" represents the secondary entity or object in a relationship' +
                                    " --> Ending of the  case: the extraction of entity relationship" },
                                     {"role": "user", "content": content1},{"role": "user", "content": content2}]
    return messages

import pandas as pd

import csv
import uuid
import os


def csv_gen(p2, file_path="./2_1_1/batch_1_1.csv", custom_id=None):
    """
    生成测试用 CSV 文件，每次输入一个测试内容时追加到文件中。

    参数:
        content_str (str): 测试内容字符串
        file_path (str): CSV 文件路径（默认：input_demo.csv）
        custom_id (str): 自定义的 custom_id（可选，若不传则自动生成）
    """
    relation_types = [
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

    attribute_types = [
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

    prompt1 = f'''
            Based on the following relation_types, extract Entities, relationships between entities  as required from the interpretation of material terms given by the next user's prompt
            Use the following relation types:
            Extracting other triples that do not contain relation types given below is prohibited
            #relation_types
            {relation_types}
            '''





    p1 = prompt1
    # 若未指定 custom_id，则生成唯一 ID（UUID 前8位）
    if custom_id is None:
        custom_id = str(uuid.uuid4())[:8]

    # 检查文件是否存在，若不存在则需要写入标题行
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 如果文件不存在，先写入标题行
        if not file_exists:
            writer.writerow(['custom_id', 'prompt1','prompt2'])

        # 写入当前测试内容的行
        writer.writerow([custom_id, p1,p2])


def csv2json():
    with open("2_1_1/batch_1_1.csv", "r") as fin:
        with open("2_1_1/batch_1_1.jsonl", 'w', encoding='utf-8') as fout:
            csvreader = csv.reader(fin)
            headers = next(csvreader)
            print("表头:", headers)
            for row in csvreader:
                body = {"model": "qwq-plus", "messages": messages_builder_example(row[1],row[2])}
                # 选择Embedding文本向量模型进行调用时，url的值需填写"/v1/embeddings"
                request = {"custom_id": row[0], "method": "POST", "url": "/v1/chat/completions", "body": body}
                fout.write(json.dumps(request, separators=(',', ':'), ensure_ascii=False) + "\n", )



def dict_get_string(dict_obj, path):
    obj = dict_obj
    try:
        for element in path:
            obj = obj[element]
        return obj
    except:
        return None

def json2csv():
    import json
    import csv
    columns = ["custom_id",
               "model",
               "request_id",
               "status_code",
               "error_code",
               "error_message",
               "created",
               "content",
               "usage"]



    with open("2_1_1/batch_ans_1_1.jsonl", "r") as fin:
        with open("2_1_1/batch_ans_1_1.csv", 'w', newline='', encoding='utf-8') as fout:
            rows = [columns]
            for line in fin:
                request_result = json.loads(line)
                row = [dict_get_string(request_result, ["custom_id"]),
                       dict_get_string(request_result, ["response", "body", "model"]),
                       dict_get_string(request_result, ["response", "request_id"]),
                       dict_get_string(request_result, ["response", "status_code"]),
                       dict_get_string(request_result, ["error", "error_code"]),
                       dict_get_string(request_result, ["error", "error_message"]),
                       dict_get_string(request_result, ["response", "body", "created"]),
                       dict_get_string(request_result, ["response", "body", "choices", 0, "message", "content"]),
                       dict_get_string(request_result, ["response", "body", "usage"])]
                rows.append(row)

            writer = csv.writer(fout)
            writer.writerows(rows)
import os
from pathlib import Path
from openai import OpenAI
import time

# 初始化客户端
client = OpenAI(
    # 若没有配置环境变量,可用百炼API Key将下行替换为：api_key="sk-xxx",但不建议在生产环境中直接将API Key硬编码到代码中,以减少API Key泄露风险.
    api_key="sk-23cc0dca396545abb1b42513f29fa3e5",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)

def upload_file(file_path):
    print(f"正在上传包含请求信息的JSONL文件...")
    file_object = client.files.create(file=Path(file_path), purpose="batch")
    print(f"文件上传成功。得到文件ID: {file_object.id}\n")
    return file_object.id

def create_batch_job(input_file_id):
    print(f"正在基于文件ID，创建Batch任务...")
    # 请注意：选择Embedding文本向量模型进行调用时,endpoint的值需填写"/v1/embeddings"
    batch = client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"Batch任务创建完成。 得到Batch任务ID: {batch.id}\n")
    return batch.id

def check_job_status(batch_id):
    print(f"正在检查Batch任务状态...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"Batch任务状态: {batch.status}\n")
    return batch.status

def get_output_id(batch_id):
    print(f"正在获取Batch任务中执行成功请求的输出文件ID...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"输出文件ID: {batch.output_file_id}\n")
    return batch.output_file_id

def get_error_id(batch_id):
    print(f"正在获取Batch任务中执行错误请求的输出文件ID...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"错误文件ID: {batch.error_file_id}\n")
    return batch.error_file_id

def download_results(output_file_id, output_file_path):
    print(f"正在打印并下载Batch任务的请求成功结果...")
    content = client.files.content(output_file_id)
    # 打印部分内容以供测试
    print(f"打印请求成功结果的前1000个字符内容: {content.text[:1000]}...\n")
    # 保存结果文件至本地
    content.write_to_file(output_file_path)
    print(f"完整的输出结果已保存至本地输出文件result.jsonl\n")

def download_errors(error_file_id, error_file_path):
    print(f"正在打印并下载Batch任务的请求失败信息...")
    content = client.files.content(error_file_id)
    # 打印部分内容以供测试
    print(f"打印请求失败信息的前1000个字符内容: {content.text[:1000]}...\n")
    # 保存错误信息文件至本地
    content.write_to_file(error_file_path)
    print(f"完整的请求失败信息已保存至本地错误文件error.jsonl\n")

def batch_llm():

    # 文件路径
    input_file_path = "2_1_1/batch_1_1.jsonl"  # 可替换为您的输入文件路径
    output_file_path = "2_1_1/batch_ans_1_1.jsonl"  # 可替换为您的输出文件路径
    error_file_path = "2_1_1/error_batch_ans_1_1.jsonl"  # 可替换为您的错误文件路径
    try:
        # Step 1_1: 上传包含请求信息的JSONL文件,得到输入文件ID,如果您需要输入OSS文件,可将下行替换为：input_file_id = "实际的OSS文件URL或资源标识符"
        input_file_id = upload_file(input_file_path)
        # Step 1_2: 基于输入文件ID,创建Batch任务
        batch_id = create_batch_job(input_file_id)
        # Step 1_3: 检查Batch任务状态直到结束
        status = ""
        while status not in ["completed", "failed", "expired", "cancelled"]:
            status = check_job_status(batch_id)
            print(f"等待任务完成...")
            time.sleep(10)  # 等待10秒后再次查询状态
        # 如果任务失败,则打印错误信息并退出
        if status == "failed":
            batch = client.batches.retrieve(batch_id)
            print(f"Batch任务失败。错误信息为:{batch.errors}\n")
            print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            return
        # Step 1_4: 下载结果：如果输出文件ID不为空,则打印请求成功结果的前1000个字符内容，并下载完整的请求成功结果到本地输出文件;
        # 如果错误文件ID不为空,则打印请求失败信息的前1000个字符内容,并下载完整的请求失败信息到本地错误文件.
        output_file_id = get_output_id(batch_id)
        if output_file_id:
            download_results(output_file_id, output_file_path)
        error_file_id = get_error_id(batch_id)
        if error_file_id:
            download_errors(error_file_id, error_file_path)
            print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")


if __name__ == "__main__":
# #step1
    df = pd.read_excel('./data/材料大辞典语料.xlsx')
    # 迷域行者
    explanation_id = 'explanation_en'
    name_id = "name_en"
    # 指定要读取的行数N
    ranges = [
    (1001, 1051),
    (2001, 2051),
    (2941, 2966),
    (4001, 4051),
    (6001, 6026)
]
# 按范围生成子列表
    num_list = []
    for start, end in ranges:
        num_list += list(range(start, end))
    print(len(num_list))
    no_extract_id = []
    for i in num_list:  # 2032报错未解决  2068, 8651  4248   2098(2100)没有英文释义   2144  {2068开始clean_space清除空格}  2599   4000, 4000  4302、5615、5821英文释义空 4000-4000 4000-8651   0-4000 4000-2933
        explanation = df.at[i, explanation_id]
        name = df.at[i, name_id]
        if (explanation.startswith("See")):
            no_extract_id.append(i)
            continue

        prompt = name + ":" + explanation
        print(prompt)
        print(f"num:{i}\n")
        csv_gen(prompt)
    with open("./2_1_1/no_extract_id.pkl", "wb") as file:  # wb 表示以二进制写模式打开文件
        pickle.dump(no_extract_id, file)
        print(no_extract_id)
    print("列表已成功存储到 'no_extract_id.pkl'")
# #step2
    csv2json()
# #step3  1_1. 准备与上传文件
    batch_llm()
# # step4
#      json2csv()

