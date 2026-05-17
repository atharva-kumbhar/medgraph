# MedGraph AI

Production-ready healthcare GraphRAG benchmark comparing three **real** inference pipelines on a 13,807-chunk medical corpus and a live TigerGraph knowledge graph (`Entity` → `RELATED_TO`).

| Pipeline | Flow |
|----------|------|
| **LLM Only** | Query → NVIDIA → answer |
| **Basic RAG** | Query → `all-MiniLM-L6-v2` embeddings → FAISS → top-k chunks → NVIDIA → answer |
| **GraphRAG** | Query → compact FAISS top-2 seed retrieval → TigerGraph traversal → keep only `usually appropriate` recommendations → rank/dedupe top 2 graph paths → compressed context → NVIDIA → answer |

No demo or mock answers. Missing API keys or TigerGraph credentials return explicit pipeline errors.

## Project layout

```text
frontend/          Dashboard (Vercel / Hugging Face static deploy)
backend/           FastAPI app, routes, services, Dockerfile, render.yaml
  medgraph/        Core pipelines (FAISS, TigerGraph, LLM, metrics)
data/              FAISS cache, ground truth, optional triplet CSVs
notebooks/         Dataset + graph build Colab export
final_medical_rag_dataset.csv
```

## Assets

- **RAG corpus:** `final_medical_rag_dataset.csv` — 13,807 chunks
- **TigerGraph:** 4,872 vertices, 5,678 `RELATED_TO` edges
- **Triplets (graph build):** `drug_triplets.csv`, `research_triplets.csv`, `clean_acr_partial.csv` → place in `data/triplets/` when rebuilding the graph (see `notebooks/`)

## Quick start (local)

### 1. Install

```powershell
cd d:\tigergraph_hk2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` at the repo root:

```powershell
copy .env.example .env
```

Required for full functionality:

```env
GEMINI_API_KEY=...          # Gemini auto-reference generation + LLM-as-a-judge
NVIDIA_API_KEY=...          # Required for default Basic RAG/GraphRAG generation
TG_HOST=https://...         # TigerGraph REST host
TG_GRAPH_NAME=database1
TG_SECRET=...               # or TG_API_TOKEN
TG_RELATIONSHIP_EDGE_TYPE=RELATED_TO
TG_MENTIONED_EDGE_TYPE=MENTIONED_IN
TG_CHUNK_VERTEX_TYPE=Chunk
MEDGRAPH_REQUEST_TIMEOUT_SECONDS=8
MEDGRAPH_GRAPH_RETRIEVAL_TOP_K=2
MEDGRAPH_GRAPH_CHUNK_LIMIT=2
TG_CHUNK_ENTITY_LIMIT=2
TG_CHUNK_FETCH_LIMIT=3
TG_VERIFY_SSL=false
TG_VERTEX_LIMIT=500
TG_MAX_EDGES=12
MEDGRAPH_LLM_ONLY_PROVIDER=nvidia
MEDGRAPH_RAG_PROVIDER=nvidia
MEDGRAPH_GRAPHRAG_PROVIDER=nvidia
```

### 3. Run backend + UI

```powershell
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The first RAG or GraphRAG query builds the FAISS index under `data/faiss_index/` (one-time, ~few minutes on CPU).

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Dataset + graph stats |
| POST | `/api/llm` | LLM-only pipeline |
| POST | `/api/rag` | FAISS RAG pipeline |
| POST | `/api/graphrag` | TigerGraph GraphRAG |
| POST | `/api/query` | All three pipelines in parallel |
| POST | `/api/metrics` | Comparison metrics (tokens, latency, cost, accuracy, graph paths) |
| POST | `/api/benchmark` | Batch evaluation |

Request body:

```json
{
  "query": "Patient has chest pain, diabetes, and kidney issues. What tests are recommended?",
  "reference_answer": "Optional override; omitted by the dashboard so Gemini auto-generates expected_answer"
}
```

Legacy aliases: `/api/llm-only`, `/api/basic-rag`.

Batch benchmark body:

```json
{
  "items": [
    {"question": "Patient has chronic chest pain and diabetes. What imaging is recommended?"},
    {"question": "Can aspirin interact with anticoagulants in a patient with kidney disease?"}
  ]
}
```

`/api/benchmark` runs LLM-only, Basic RAG, and GraphRAG for every question, auto-generates `expected_answer`, evaluates each answer with Gemini PASS/FAIL plus BERTScore F1, and returns average metrics plus GraphRAG-vs-RAG improvements.

## Deploy backend (Render)

1. Connect this repo to [Render](https://render.com).
2. Use `backend/render.yaml` or create a **Web Service** with:
   - **Dockerfile path:** `backend/Dockerfile`
   - **Docker context:** repository root
3. Set environment variables from `backend/.env.example` in the Render dashboard.
4. Note the public URL (e.g. `https://medgraph-api.onrender.com`).

## Deploy frontend (Vercel)

1. Set root directory to `frontend/`.
2. Before `config.js`, inject your Render API URL:

```html
<script>window.MEDGRAPH_API_BASE = "https://medgraph-api.onrender.com";</script>
```

3. Set `MEDGRAPH_CORS_ORIGINS` on Render to your Vercel domain.

See `frontend/README.md` for Hugging Face static hosting.

## Docker (local)

```powershell
docker build -f backend/Dockerfile -t medgraph-api .
docker run --env-file .env -p 8000:8000 medgraph-api
```

## Evaluation

- **Tokens / cost:** from hosted LLM usage when available; otherwise estimated
- **Latency:** measured wall-clock per pipeline
- **Accuracy:** Gemini auto-generates `expected_answer`, then Gemini LLM-as-a-Judge returns PASS/FAIL and BERTScore F1 compares each answer to that reference (if `bert-score` is installed)
- **Graph paths:** live TigerGraph `RELATED_TO` entity traversal; `relationship` is stored as an edge attribute. Graph chunks are retrieved through the installed TigerGraph `get_chunks(entity_name)` query endpoint.
- **GraphRAG efficiency:** graph traversal is skipped for simple factual lookups, recommendation paths are filtered to `usually appropriate`, redundant procedures are deduplicated, and the LLM receives a short “Top medical recommendations” context.
- **Batch Evaluation page:** paste 50-60 questions, run all pipelines, view average tokens/latency/cost/judge pass %/BERTScore, calculate GraphRAG token and cost reduction vs Basic RAG, and export per-question results to CSV.

## Research benchmark only

This tool is for educational benchmarking. It is **not** medical advice.
