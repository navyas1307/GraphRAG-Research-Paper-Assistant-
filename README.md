# GraphRAG Research Paper Assistant

An AI-powered research paper management system that combines **Vector Search (RAG)** with **Knowledge Graphs (GraphRAG)** to enable intelligent retrieval, comparison, and analysis of academic papers.

---

## Overview

This system allows users to upload research papers and automatically:

* Extract structured content (sections, figures, metadata)
* Generate embeddings for semantic search
* Build a knowledge graph of entities (methods, datasets, topics)
* Perform hybrid similarity using graph + vector signals
* Enable explainable comparisons and recommendations

---

## Key Features

### Paper Processing

* PDF parsing using GROBID / pdfplumber 
* Section extraction, figures, and metadata
* Immediate keyword-based entity extraction

### Semantic Search (RAG)

* Sentence embeddings using BGE model 
* Stored in Qdrant vector database 
* Chunk-level retrieval for precise answers

### Knowledge Graph (GraphRAG)

* Built using Neo4j 
* Nodes: Paper, Method, Dataset, Model, Topic
* Edges: semantic relationships (USES_METHOD, HAS_TOPIC, etc.)

### Hybrid Similarity Engine

* Combines:

  * Vector similarity (cosine)
  * Graph-based entity overlap
* Cross-domain suppression + semantic boost 

### LLM Integration

* Gemini / Ollama fallback system 
* Generates:

  * Research summaries
  * Gap analysis
  * Comparisons (GraphRAG-first prompting)

### Evaluation System

* Automatic evaluation pipeline:

  * Precision@K, MRR, nDCG
  * Graph signal coverage
  * Explainability metrics 


---

## Tech Stack

### Backend

* FastAPI 
* PostgreSQL 
* Neo4j (graph DB)
* Qdrant (vector DB)
* Sentence Transformers

### Frontend

* Next.js 14 
* Tailwind CSS 
* D3.js (graph visualization)
* Recharts (analytics)

---

## System Workflow

1. Upload PDF
2. Extract text + sections
3. Generate embeddings
4. Extract entities (LLM + keyword fallback)
5. Build Neo4j knowledge graph
6. Create similarity edges
7. Cluster papers
8. Enable search, comparison, recommendations

---

## Evaluation Metrics

* Similarity Improvement (Hybrid vs Baseline)
* Cross-domain suppression effectiveness
* Explainability precision
* Graph signal coverage

---

## Project Structure

Backend:

* `main.py` → FastAPI app
* `pdf_processor.py` → PDF extraction
* `embeddings.py` → vector generation
* `graph_db.py` → Neo4j graph
* `similarity_model.py` → hybrid scoring
* `graphrag_compare.py` → GraphRAG comparison

Frontend:

* Dashboard (chat interface)
* Library (paper management)
* Graph view (knowledge graph)
* Compare (GraphRAG comparison)
* Evaluation dashboard

---

## How to Run

###

```bash
docker compose up -d
docker ps
```

### Backend

```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Key Innovation

Unlike traditional RAG systems, this project:

* Uses **graph signals as primary reasoning input**
* Ensures **explainability via entity overlap**
* Prevents hallucination using structured graph context
* Combines **semantic + relational intelligence**

---

## Future Improvements

* Real-time arXiv integration
* Advanced entity linking (cross-paper resolution)
* Multi-hop reasoning over graphs
* Better clustering for large corpora

---

## Author

Navya Sharma
(Data + AI Systems | GraphRAG Research)
