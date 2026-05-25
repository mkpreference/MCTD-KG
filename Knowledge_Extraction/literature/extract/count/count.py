import json
import csv
from collections import Counter

csv_path = "/root/files_lpz/ra/实验/literature/extract/ans/batch_ans_literature_chunks.csv"

entity_set = set()
relation_triples = 0
attribute_triples = 0
relation_counter = Counter()
attribute_counter = Counter()

def safe_add(obj_set, val):
    if isinstance(val, dict):
        val = json.dumps(val, ensure_ascii=False, sort_keys=True)
    elif not isinstance(val, str):
        val = str(val)
    obj_set.add(val)

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        content_str = row.get("content", "")
        if not content_str:
            continue
        try:
            triples = json.loads(content_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(triples, list):
            continue
        for t in triples:
            if not isinstance(t, dict):
                continue
            if "head" in t and "relation" in t:
                relation_triples += 1
                safe_add(entity_set, t["head"])
                safe_add(entity_set, t["tail"])
                relation_counter[t["relation"]] += 1
            elif "entity" in t and "attribute" in t:
                attribute_triples += 1
                safe_add(entity_set, t["entity"])
                attribute_counter[t["attribute"]] += 1

print(f"Entities (unique): {len(entity_set)}")
print(f"Relation triples: {relation_triples}")
print(f"Attribute triples: {attribute_triples}")
print(f"\nRelation types:")
for rel, cnt in relation_counter.most_common():
    print(f"  {rel}: {cnt}")
print(f"\nAttribute types:")
for attr, cnt in attribute_counter.most_common():
    print(f"  {attr}: {cnt}")
