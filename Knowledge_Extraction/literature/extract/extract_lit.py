import os
import json
import uuid
import csv

# 复用您之前的 messages_builder_example 函数（包含 11种关系 和 12种属性的定义）
from batch_prompt import messages_builder_example 

def process_literature_chunks(base_dir, output_jsonl_path):
    """遍历文献chunks目录，生成百炼Batch API所需的JSONL文件"""
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        # 遍历根目录下的所有分类文件夹 (例如 Biomedical Materials)
        for category_folder in os.listdir(base_dir):
            category_path = os.path.join(base_dir, category_folder)
            if not os.path.isdir(category_path):
                continue
                
            # 遍历该分类下的所有 JSON 文件
            for file_name in os.listdir(category_path):
                if not file_name.endswith('.json'):
                    continue
                    
                file_path = os.path.join(category_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        chunks_data = json.load(f)
                        # 假设您的 JSON 是一个列表，或者包含文本块的字典
                        # 这里假设 chunks_data 是一个包含 {"text": "..."} 的列表
                        for chunk in chunks_data:
                            text_content = chunk.get("text", "")
                            if len(text_content) < 50: # 过滤掉极短的噪音块
                                continue
                            
                            # 构造 LLM Prompt
                            # 为了保持上下文，可以将“所属分类”作为提示词的一部分
                            user_prompt_1 = f"Please extract knowledge from the following literature text belonging to the category: {category_folder}."
                            user_prompt_2 = text_content
                            
                            messages = messages_builder_example(user_prompt_1, user_prompt_2)
                            
                            # 生成带有溯源信息的 custom_id (分类名_文件名_uuid)
                            doc_id = file_name.replace('_chunks.json', '')
                            custom_id = f"{category_folder}@@{doc_id}@@{str(uuid.uuid4())[:8]}"
                            
                            body = {
                                "model": "qwen-max", # 建议文献抽取使用 max 以处理复杂从句
                                "messages": messages
                            }
                            request = {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
                            fout.write(json.dumps(request, separators=(',', ':'), ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

# 执行生成
process_literature_chunks("/root/files_lpz/ra/实验/literature/literature_data/3_llm_chunks", "/root/files_lpz/ra/实验/literature/literature_data/literature_batch_tasks.jsonl")