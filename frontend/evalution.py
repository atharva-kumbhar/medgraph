import pandas as pd
import time
import random

# -------------------------
# Replace these with your actual functions
# -------------------------

def run_llm(query):
    time.sleep(1)
    answer = "LLM answer"
    tokens = random.randint(200, 400)
    cost = tokens * 0.0000008
    return answer, tokens, cost

def run_rag(query):
    time.sleep(2)
    answer = "RAG answer"
    tokens = random.randint(500, 900)
    cost = tokens * 0.0000008
    return answer, tokens, cost

def run_graphrag(query):
    time.sleep(2)
    answer = "GraphRAG answer"
    tokens = random.randint(250, 600)
    cost = tokens * 0.0000008
    return answer, tokens, cost

def generate_expected_answer(query):
    return "Expected answer"

def bert_score(predicted, expected):
    return round(random.uniform(0.78, 0.92), 4)

def llm_judge(predicted, expected):
    return random.choice(["PASS", "FAIL"])


# -------------------------
# Load questions
# -------------------------

with open("queries.txt", "r", encoding="utf-8") as f:
    queries = [q.strip() for q in f.readlines() if q.strip()]

results = []

for i, query in enumerate(queries):
    print(f"Running {i+1}/{len(queries)}: {query}")

    expected = generate_expected_answer(query)

    # LLM
    start = time.time()
    llm_answer, llm_tokens, llm_cost = run_llm(query)
    llm_latency = round(time.time() - start, 2)

    # RAG
    start = time.time()
    rag_answer, rag_tokens, rag_cost = run_rag(query)
    rag_latency = round(time.time() - start, 2)

    # GraphRAG
    start = time.time()
    graph_answer, graph_tokens, graph_cost = run_graphrag(query)
    graph_latency = round(time.time() - start, 2)

    results.append({
        "query": query,

        "llm_tokens": llm_tokens,
        "rag_tokens": rag_tokens,
        "graph_tokens": graph_tokens,

        "llm_latency": llm_latency,
        "rag_latency": rag_latency,
        "graph_latency": graph_latency,

        "llm_cost": llm_cost,
        "rag_cost": rag_cost,
        "graph_cost": graph_cost,

        "llm_judge": llm_judge(llm_answer, expected),
        "rag_judge": llm_judge(rag_answer, expected),
        "graph_judge": llm_judge(graph_answer, expected),

        "llm_bert": bert_score(llm_answer, expected),
        "rag_bert": bert_score(rag_answer, expected),
        "graph_bert": bert_score(graph_answer, expected),
    })


# -------------------------
# Save CSV
# -------------------------

df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)

print("\nCSV saved as benchmark_results.csv")


# -------------------------
# Final averages
# -------------------------

print("\n===== FINAL AVERAGES =====")

print("Avg LLM Tokens:", df["llm_tokens"].mean())
print("Avg RAG Tokens:", df["rag_tokens"].mean())
print("Avg Graph Tokens:", df["graph_tokens"].mean())

print("Avg LLM Latency:", df["llm_latency"].mean())
print("Avg RAG Latency:", df["rag_latency"].mean())
print("Avg Graph Latency:", df["graph_latency"].mean())

print("Avg LLM Cost:", df["llm_cost"].mean())
print("Avg RAG Cost:", df["rag_cost"].mean())
print("Avg Graph Cost:", df["graph_cost"].mean())

print("Avg LLM BERT:", df["llm_bert"].mean())
print("Avg RAG BERT:", df["rag_bert"].mean())
print("Avg Graph BERT:", df["graph_bert"].mean())


# -------------------------
# Graph improvements
# -------------------------

token_reduction = (
    (df["rag_tokens"].mean() - df["graph_tokens"].mean())
    / df["rag_tokens"].mean()
) * 100

cost_reduction = (
    (df["rag_cost"].mean() - df["graph_cost"].mean())
    / df["rag_cost"].mean()
) * 100

print(f"\nGraph Token Reduction vs RAG: {token_reduction:.2f}%")
print(f"Graph Cost Reduction vs RAG: {cost_reduction:.2f}%")