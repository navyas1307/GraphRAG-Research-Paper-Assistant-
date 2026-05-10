"""
entity_normalizer.py — v3
────────────────────────────────────────────────────────────────────────────────
Root cause of 0% normalization hit rate (fixed here)
──────────────────────────────────────────────────────
LLM-extracted entities rarely match synonym keys exactly.
Examples of what LLM actually produces vs what was in the map:
  "RAG-based approach"           ← not in map (only "rag" exactly)
  "large language models (LLMs)" ← not in map (only "large language models")
  "Transformer architecture"     ← only "transformer architecture" (lowercase)
  "fine tuning the model"        ← not in map

Fix: four-stage matching pipeline
  1. Exact lowercase match
  2. Strip trailing parentheticals, retry
  3. Longest-prefix match
  4. Substring containment — entity string CONTAINS a known key
     e.g. "rag-based retrieval" contains "rag" → canonical "rag"
     e.g. "fine tuning the model" contains "fine tuning" → "fine-tuning"

Stage 4 is what was missing and causes most hits in practice.
"""

from __future__ import annotations

import os, re
from typing import Dict, List, Set, Tuple

SYNONYM_MAP: Dict[str, str] = {
    # RAG
    "retrieval augmented generation (rag)":    "rag",
    "retrieval-augmented generation (rag)":    "rag",
    "retrieval augmented generation":          "rag",
    "retrieval-augmented generation":          "rag",
    "retrieval augmentation":                  "rag",
    "retrieval-augmented":                     "rag",
    "rag":                                     "rag",
    "graph rag":                               "graphrag",
    "graph-rag":                               "graphrag",
    "graphrag":                                "graphrag",
    # LLMs
    "large language model (llm)":              "llm",
    "large language models (llms)":            "llm",
    "large language model":                    "llm",
    "large language models":                   "llm",
    "llms":                                    "llm",
    "llm":                                     "llm",
    # Transformers
    "transformer model":                       "transformer",
    "transformer models":                      "transformer",
    "transformer architecture":                "transformer",
    "transformer-based":                       "transformer",
    "transformer":                             "transformer",
    "attention mechanism":                     "attention",
    "self-attention":                          "attention",
    "self attention":                          "attention",
    "multi-head attention":                    "attention",
    "multi head attention":                    "attention",
    # Fine-tuning
    "fine tuning":                             "fine-tuning",
    "finetuning":                              "fine-tuning",
    "fine-tuning":                             "fine-tuning",
    "full fine-tuning":                        "fine-tuning",
    "instruction fine-tuning":                 "instruction fine-tuning",
    "instruction tuning":                      "instruction fine-tuning",
    "parameter efficient fine-tuning":         "peft",
    "parameter-efficient fine-tuning":         "peft",
    "peft":                                    "peft",
    "lora":                                    "lora",
    "low-rank adaptation":                     "lora",
    "low rank adaptation":                     "lora",
    # BERT
    "bert-based model":                        "bert",
    "bert model":                              "bert",
    "bert":                                    "bert",
    "bidirectional encoder representations":   "bert",
    "roberta":                                 "roberta",
    "deberta":                                 "deberta",
    # GPT family
    "generative pre-trained transformer":      "gpt",
    "gpt model":                               "gpt",
    "gpt":                                     "gpt",
    "chatgpt":                                 "chatgpt",
    "gpt-4":                                   "gpt-4",
    "gpt4":                                    "gpt-4",
    "gpt-3":                                   "gpt-3",
    "llama":                                   "llama",
    "llama-2":                                 "llama-2",
    "llama 2":                                 "llama-2",
    "llama2":                                  "llama-2",
    "llama-3":                                 "llama-3",
    "llama 3":                                 "llama-3",
    "mistral":                                 "mistral",
    # Datasets
    "squad 1.1":                               "squad",
    "squad 2.0":                               "squad",
    "squad":                                   "squad",
    "ms marco":                                "msmarco",
    "ms-marco":                                "msmarco",
    "msmarco":                                 "msmarco",
    "natural questions (nq)":                  "natural questions",
    "natural questions":                       "natural questions",
    "nq":                                      "natural questions",
    "trivia qa":                               "triviaqa",
    "hotpot qa":                               "hotpotqa",
    # Tasks
    "question answering (qa)":                 "question answering",
    "open-domain question answering":          "question answering",
    "open domain qa":                          "question answering",
    "question answering":                      "question answering",
    "qa":                                      "question answering",
    "named entity recognition (ner)":          "named entity recognition",
    "ner":                                     "named entity recognition",
    "natural language processing (nlp)":       "nlp",
    "natural language processing":             "nlp",
    "nlp":                                     "nlp",
    "information retrieval (ir)":              "information retrieval",
    "information retrieval":                   "information retrieval",
    "ir":                                      "information retrieval",
    "summarization":                           "summarization",
    "text summarization":                      "summarization",
    "machine translation":                     "machine translation",
    # Methods
    "dense passage retrieval (dpr)":           "dense passage retrieval",
    "dense retrieval":                         "dense passage retrieval",
    "dpr":                                     "dense passage retrieval",
    "bm25":                                    "bm25",
    "tf-idf":                                  "tf-idf",
    "tfidf":                                   "tf-idf",
    "in-context learning":                     "in-context learning",
    "in context learning":                     "in-context learning",
    "icl":                                     "in-context learning",
    "chain-of-thought (cot)":                  "chain-of-thought",
    "chain of thought":                        "chain-of-thought",
    "cot":                                     "chain-of-thought",
    "rlhf":                                    "rlhf",
    "reinforcement learning from human feedback": "rlhf",
    "dpo":                                     "dpo",
    "direct preference optimization":          "dpo",
    "contrastive learning":                    "contrastive learning",
    "knowledge distillation":                  "knowledge distillation",
    "prompting":                               "prompting",
    "prompt tuning":                           "prompting",
    "few-shot prompting":                      "prompting",
    "zero-shot prompting":                     "prompting",
    "few shot":                                "prompting",
    "zero shot":                               "prompting",
    "vector database":                         "vector database",
    "vector store":                            "vector database",
    "embeddings":                              "embeddings",
    "sentence embeddings":                     "embeddings",
    "word embeddings":                         "embeddings",
    # Topics
    "knowledge graphs":                        "knowledge graph",
    "knowledge graph":                         "knowledge graph",
    "gnn":                                     "gnn",
    "graph neural network":                    "gnn",
    "graph neural networks":                   "gnn",
    "deep learning":                           "deep learning",
    "machine learning":                        "machine learning",
    "ml":                                      "machine learning",
    "dl":                                      "deep learning",
    "neural network":                          "neural network",
    "neural networks":                         "neural network",
    "reinforcement learning":                  "reinforcement learning",
    "rl":                                      "reinforcement learning",
    "computer vision":                         "computer vision",
    "speech recognition":                      "speech recognition",
    "multimodal":                              "multimodal ai",
    "multi-modal":                             "multimodal ai",
    "multimodal learning":                     "multimodal ai",
    "federated learning":                      "federated learning",
    "seq2seq":                                 "seq2seq",
    "sequence to sequence":                    "seq2seq",
    "mixture of experts":                      "mixture of experts",
    "moe":                                     "mixture of experts",
}

_SORTED_KEYS: List[str] = sorted(SYNONYM_MAP.keys(), key=len, reverse=True)
_PAREN_RE    = re.compile(r"\s*\(.*?\)\s*$")
_DEBUG       = os.getenv("ENTITY_NORM_DEBUG", "0") == "1"


def _clean(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower())


def _apply_synonyms(s: str) -> str:
    # 1. Exact
    if s in SYNONYM_MAP:
        return SYNONYM_MAP[s]
    # 2. Strip parens then exact
    stripped = _PAREN_RE.sub("", s).strip()
    if stripped != s and stripped in SYNONYM_MAP:
        return SYNONYM_MAP[stripped]
    # 3. Prefix match
    for key in _SORTED_KEYS:
        if s.startswith(key):
            rem = s[len(key):].strip()
            if not rem or re.match(r"^\(.*\)$", rem):
                return SYNONYM_MAP[key]
    # 4. Substring containment (KEY FIX — catches "rag-based", "fine tuning the model", etc.)
    for key in _SORTED_KEYS:
        if len(key) >= 3 and key in s:  # min 3 chars to avoid spurious matches
            return SYNONYM_MAP[key]
    return s


def normalize_entity(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    return _apply_synonyms(_clean(raw))


def normalize_entity_list(entities: List[str], deduplicate: bool = True) -> List[str]:
    normalized = [normalize_entity(e) for e in entities if e and e.strip()]
    if deduplicate:
        seen: Set[str] = set()
        result = []
        for e in normalized:
            if e and e not in seen:
                seen.add(e); result.append(e)
        return result
    return [e for e in normalized if e]


def normalize_entity_list_debug(entities: List[str], label: str = "") -> Tuple[List[str], int]:
    normalized, hits = [], 0
    for raw in entities:
        if not raw or not raw.strip():
            continue
        norm = normalize_entity(raw)
        if norm != _clean(raw):
            hits += 1
            if _DEBUG:
                print(f"  [NORM{' '+label if label else ''}] '{raw}' → '{norm}'")
        normalized.append(norm)
    if _DEBUG and entities:
        print(f"  [NORM] {label}: {hits}/{len(entities)} synonym hits")
    seen: Set[str] = set()
    result = []
    for e in normalized:
        if e and e not in seen:
            seen.add(e); result.append(e)
    return result, hits


def normalize_paper_entities(paper: dict, debug: bool = False) -> dict:
    entities = paper.get("entities") or {}
    pid      = paper.get("paper_id", "?")[:8]
    total    = 0

    if debug or _DEBUG:
        print(f"\n[NORM] Normalizing paper {pid}…")

    for field in ("methods", "datasets", "models", "tasks", "topics"):
        raw = entities.get(field) or []
        if raw:
            normed, hits = normalize_entity_list_debug(raw, f"{field}@{pid}")
            entities[field] = normed
            total += hits
        else:
            entities[field] = []

    paper["entities"] = entities

    raw_topics = paper.get("topics") or []
    if raw_topics:
        normed, hits = normalize_entity_list_debug(raw_topics, f"topics@{pid}")
        paper["topics"] = normed
        total += hits
    else:
        paper["topics"] = []

    if debug or _DEBUG:
        print(f"[NORM] Paper {pid}: {total} synonym hits total")

    return paper


def check_overlap_debug(paper_a: dict, paper_b: dict) -> dict:
    normalize_paper_entities(paper_a)
    normalize_paper_entities(paper_b)

    def _s(p, fld, top=False):
        raw = p.get("topics", []) if top else (p.get("entities") or {}).get(fld, []) or []
        s = {normalize_entity(e) for e in raw}; s.discard(""); return s

    mo = sorted(_s(paper_a,"methods") & _s(paper_b,"methods"))
    do = sorted(_s(paper_a,"datasets") & _s(paper_b,"datasets"))
    to = sorted(_s(paper_a,"topics",True) & _s(paper_b,"topics",True))

    if _DEBUG:
        print(f"\n[OVERLAP] {paper_a.get('paper_id','A')[:8]} ↔ {paper_b.get('paper_id','B')[:8]}")
        print(f"  Methods overlap:  {mo}")
        print(f"  Datasets overlap: {do}")
        print(f"  Topics overlap:   {to}")

    return {"methods_overlap": mo, "datasets_overlap": do, "topics_overlap": to,
            "has_signal": bool(mo or do or to)}