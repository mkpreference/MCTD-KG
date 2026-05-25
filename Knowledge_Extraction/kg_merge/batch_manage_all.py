# coding:utf-8
import codecs
import os
import shutil
try:
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())
except ImportError:
    pass
import json
import csv
# from graph import build_graph  # not available
import pandas as pd
import pickle
import datetime


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

    def get_completion(self, prompt,num):

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
        print("///////////")
        print(extracted_relations)
        try:
            extracted_jsons = json.loads(extracted_relations)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")


        extracted_jsons = self.clean_spaces(extracted_jsons)
        print(extracted_jsons)
        return extracted_jsons
      #删除每个字符串头尾llm可能产生的空格


def parse_custom_id(custom_id):
    """
    Parse custom_id like 'Category__LiteratureName__chunk0000'
    Returns (category, literature_name, chunk_idx) or None if parsing fails.
    """
    parts = custom_id.split('__')
    if len(parts) < 3:
        return None
    category = parts[0]
    chunk_part = parts[-1]  # e.g. 'chunk0000'
    literature_name = '__'.join(parts[1:-1])
    try:
        chunk_idx = int(chunk_part.replace('chunk', ''))
    except ValueError:
        chunk_idx = -1
    return category, literature_name, chunk_idx


if __name__ == "__main__":
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

    # ----- Input / Output paths -----
    ans_dir = "/root/files_lpz/ra/实验/literature/extract/ans"
    input_csv = os.path.join(ans_dir, "batch_ans_literature_chunks.csv")
    output_dir = "/root/files_lpz/ra/实验/literature/extract/merge/parsed_results"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # Collect parsed results grouped by category -> literature -> chunks
    all_results = {}          # custom_id -> parsed JSON
    by_category = {}          # category -> literature -> list of (chunk_idx, parsed_json)
    success_count = 0
    error_count = 0
    skip_count = 0

    for i, row in df.iterrows():
        custom_id = row["custom_id"]
        status_code = row["status_code"]
        content = row["content"]

        parsed = parse_custom_id(custom_id)
        if parsed is None:
            print(f"[SKIP] Cannot parse custom_id: {custom_id}")
            skip_count += 1
            continue
        category, lit_name, chunk_idx = parsed

        if status_code != 200:
            print(f"[SKIP] status={status_code} for {custom_id}")
            skip_count += 1
            continue

        if pd.isna(content) or not str(content).strip():
            print(f"[SKIP] empty content for {custom_id}")
            skip_count += 1
            continue

        material_ = material()
        try:
            ans = material_.get_completion(str(content), chunk_idx)
        except Exception as e:
            print(f"[ERROR] parsing {custom_id}: {e}")
            error_count += 1
            continue

        all_results[custom_id] = ans
        by_category.setdefault(category, {}).setdefault(lit_name, []).append((chunk_idx, ans))
        success_count += 1

        if success_count % 50 == 0:
            print(f"  ... processed {success_count} entries")

    print(f"\nDone. success={success_count}, errors={error_count}, skipped={skip_count}")

    # Save per-category results
    for category, lit_dict in by_category.items():
        safe_cat = category.replace(' ', '_').replace('/', '_')
        cat_file = os.path.join(output_dir, f"{safe_cat}.json")
        sorted_lit_dict = {}
        for lit_name, chunks in lit_dict.items():
            sorted_lit_dict[lit_name] = [ans for _, ans in sorted(chunks, key=lambda x: x[0])]
        with open(cat_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_lit_dict, f, ensure_ascii=False, indent=2)
        total_chunks = sum(len(v) for v in sorted_lit_dict.values())
        print(f"  Saved {safe_cat}.json: {len(sorted_lit_dict)} papers, {total_chunks} parsed chunks")

    # Save flat all-in-one file
    all_file = os.path.join(output_dir, "all_parsed.json")
    with open(all_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved all_parsed.json: {len(all_results)} total entries")
