# MCTD-KG: Materials Commonsense Ternary-question Knowledge Graph

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-4.4-green.svg)](https://neo4j.com/)

> **Towards Building a Multi-source Heterogeneous Knowledge Graph for Complex Material Question Answering**

MCTD-KG is a multi-source, heterogeneous knowledge graph for materials science that integrates knowledge from scientific literature, materials toolbooks, and structured databases. It supports complex material question answering through a hybrid retrieval-augmented generation pipeline combining knowledge-graph retrieval (KGQA), dense retrieval (RAG), and LLM reasoning.

---

## Overview

Materials science knowledge is scattered across heterogeneous sources — scientific papers describe experimental findings, toolbooks catalogue standardized properties, and databases encode structured facts. MCTD-KG unifies these into a single knowledge graph with 12 relation types, 12+ material domains, and a ternary-question QA benchmark for evaluating multi-hop reasoning.

### Key Features

- **Multi-source Knowledge Extraction**: LLM-driven extraction pipeline covering scientific literature (PDF → markdown → chunks → triples) and structured toolbook data
- **Entity Linking & KG Merging**: Fuzzy matching + LLM-based entity alignment to resolve synonymous terms across sources
- **12 Standardized Relations**: `subclassOf`, `isAKindOf`, `isRelatedTo`, `isPartOf`, `isUsedFor`, `isMadeOf`, `isDerivedFrom`, `isTestedBy`, `isSimilarTo`, `isComplementaryTo`, `isDependOn`, `isAlternativeTo`
- **MCTD-QA Benchmark**: Ternary-question QA dataset with 1-hop to 3-hop reasoning chains, each question annotated with gold answer, relation path, and required terms
- **Comprehensive Evaluation**: KGQA, RAG (top-k retrieval), pure LLM baselines, model-scale comparison (Llama-3.1-8B, Llama-3.2-3B, Qwen3-4B, Qwen3-8B), and KG-scale ablation

---

## Project Structure

```
MCTD-KG/
├── IE_dataset/                        # Information Extraction dataset & evaluation
│   ├── IE_dataset_100.json            #   IE benchmark (100 samples)
│   ├── IE_dataset_200.json            #   IE benchmark (200 samples)
│   ├── eval_code/                     #   Evaluation scripts (batch prompting)
│   │   ├── batch_prompt_1_*.py        #     Prompt strategy 1: direct extraction
│   │   ├── batch_prompt_2_*_关系.py   #     Prompt strategy 2: relation extraction
│   │   ├── batch_prompt_2_*_属性.py   #     Prompt strategy 2: attribute extraction
│   │   └── graph.py                   #     Neo4j graph construction utility
│   └── eval_result/                   #   Evaluation outputs (CSV, JSONL, metrics)
│
├── QA-dataset/                        # MCTD-QA: Ternary-question QA benchmark
│   ├── qa_dataset.json                #   QA pairs (1-hop, 2-hop, 3-hop questions)
│   ├── download_bge_m3.sh             #   Embedding model download script
│   ├── eval_code/
│   │   ├── kgqa.py                    #   Hybrid KGQA pipeline (KG + LLM)
│   │   ├── rag_qa_top{5,8,10,15}.py   #   RAG baselines with varying top-k
│   │   ├── rag_topk.py                #   RAG top-k search utility
│   │   ├── llm_qa.py                  #   Pure LLM baseline (no retrieval)
│   │   ├── evaluate_pure_llm.py       #   Evaluation metrics
│   │   ├── kg_scale/                  #   KG-scale ablation (13%/23%/33%/tool/lit)
│   │   └── modevs/                    #   Model-scale comparison (8B/4B/3B)
│   └── eval_reasult/                  #   Evaluation results (JSON)
│
├── Knowledge_Extraction/              # KG construction pipeline
│   ├── literature/                    #   Scientific literature extraction
│   │   ├── pdf_parse.py               #     PDF → markdown converter
│   │   ├── pdf_parse1.py              #     Alternative PDF parser
│   │   ├── extract/                   #     LLM triple extraction from chunks
│   │   │   ├── extract_lit.py         #       Batch extraction coordinator
│   │   │   ├── batch_prompt.py        #       LLM prompting for triples
│   │   │   └── count/count.py         #       Triple statistics
│   │   └── requirements.txt           #     Python dependencies
│   ├── toolbook/                      #   Toolbook/materials encyclopedia extraction
│   │   ├── batch_manage.py            #     Batch extraction manager
│   │   └── batch_prompt.py            #     LLM prompting for toolbook triples
│   ├── kg_merge/                      #   KG merging & entity linking
│   │   ├── entity_linking.py          #     Entity alignment (fuzzy + LLM)
│   │   ├── batch_manage_all.py        #     Batch merge coordinator
│   │   ├── optimize_promoted_terms.py #     Term promotion optimization
│   │   ├── calculate_other2term_threshold.py  # Synonym threshold calibration
│   │   ├── add_neo4j.py               #     Neo4j graph import
│   │   ├── test_entity_linking.py     #     Unit tests
│   │   └── parsed_results/            #     Extracted KG triples per domain
│   └── ex_tri_count/                  #   Triple count statistics
│       ├── coun_literature.py
│       └── coun_toolbook.py
│
├── .gitignore
└── README.md
```

---

## IE Dataset

The Information Extraction dataset (`IE_dataset/`) provides 100 and 200 annotated material science paragraphs for evaluating LLM-based triple extraction:

- **Relations**: Entity-relation-entity triples (`head`, `relation`, `tail`)
- **Attributes**: Entity-attribute-value triples (`entity`, `attribute`, `value`)
- **Evaluation**: Multiple prompting strategies benchmarked with batch processing

---
## QA Dataset

The QA dataset (`QA-dataset/qa_dataset.json`) contains ternary questions across 12 material domains. Each question is annotated with:

| Field | Description |
|-------|-------------|
| `id` | Unique question identifier |
| `top_category` | Material domain (e.g., Composites, Metals, Nanomaterials) |
| `question_en` / `question_zh` | Bilingual question text |
| `gold_answer_en` / `gold_answer_zh` | Ground-truth answer |
| `hop_count` | Reasoning steps required (1, 2, or 3) |
| `relations_path` | Expected relation chain for multi-hop reasoning |
| `required_terms` | Key entities needed to answer the question |
| `strategy_tested` | Evaluation strategy label |
| `source` | Data provenance (`toolbook`, `literature`) |

### Question Types by Hop Count

- **1-hop**: Direct fact retrieval (e.g., "Which category does X belong to?")
- **2-hop**: Two-step reasoning chains (e.g., "What material in category Y serves as the binding substance in X?")
- **3-hop**: Complex multi-step reasoning across domains

---
## Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/mkpreference/MCTD-KG.git
cd MCTD-KG
pip install -r requirements.txt
```

### 2. Set up Neo4j

Download and install [Neo4j 4.4](https://neo4j.com/download/) or later. Update connection settings in the scripts:

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

### 3. Configure LLM API

Set your OpenAI-compatible API credentials (e.g., DashScope, DeepSeek):

```python
API_KEY = "your_api_key"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"
```

### 4. Download embedding model (for KGQA)

```bash
cd QA-dataset
bash download_bge_m3.sh
```

---

## Usage

### QA Evaluation

```bash
cd QA-dataset/eval_code

# KGQA pipeline
python kgqa.py

# RAG baselines (varying top-k)
python rag_qa_top5.py
python rag_qa_top10.py

# Pure LLM baseline
python llm_qa.py
python evaluate_pure_llm.py
```

---

## Key Components

### Hybrid KGQA Pipeline (`kgqa.py`)

Combines three retrieval strategies:
1. **Graph retrieval**: Cypher queries to traverse relations in Neo4j
2. **Semantic retrieval**: Dense embedding matching via BGE-M3 for entity disambiguation
3. **LLM reasoning**: Qwen-Max for final answer synthesis with chain-of-thought

### Entity Linking (`entity_linking.py`)

Two-stage entity alignment:
- **Stage 1**: `rapidfuzz` fuzzy matching to identify candidate synonym pairs
- **Stage 2**: LLM-based semantic verification to confirm or reject merges, with detailed reasoning logs
---

## Results Summary

Evaluation results are organized under `QA-dataset/eval_reasult/`:

| Evaluation | Directory | Purpose |
|------------|-----------|---------|
| KGQA | `qa_ans/` | Full KG-augmented QA performance |
| RAG top-k | `qa_ans/rag_eval_results/` | Retrieval quality vs. top-k |
| Pure LLM | `qa_ans/` | Baseline without KG |
| KG scale ablation | `kg_scale/` | Varying KG coverage (13%–100%) |
| Model scale | `modevs/` | Open-source model comparison |

---

## Material Domains Covered

Biomedical Materials, Composites, Energy Materials, Fundamental of Material Science, Information Materials, Inorganic Non-metallic Materials, Metals, Nanomaterials, Natural Materials and Products, Organic Polymer Materials, Special Functional Materials

---

## Relation Ontology

```
subclassOf  •  isAKindOf  •  isRelatedTo  •  isPartOf
isUsedFor   •  isMadeOf   •  isDerivedFrom •  isTestedBy
isSimilarTo •  isComplementaryTo •  isDependOn •  isAlternativeTo
```

---

## Citation

If you use MCTD-KG in your research, please cite the corresponding paper.

---

