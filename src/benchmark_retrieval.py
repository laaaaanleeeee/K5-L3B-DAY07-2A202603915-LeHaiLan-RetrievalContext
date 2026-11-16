import json, os, statistics, sys, time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrievers import (
    LexicalTfidfRetriever,
    DenseVectorRetriever,
    HybridRRFRetriever,
    pack_context
)

def hit(chunk: Dict[str, Any], gold: List[Dict[str, Any]]) -> bool:
    """Checks if chunk matches any gold item (same source_id, same page, and evidence in text)."""
    return any(
        chunk["source_id"] == g["source_id"]
        and chunk["page"] == g["page"]
        and g["evidence"] in chunk["text"]
        for g in gold
    )

def evaluate_retriever(
    name: str,
    retriever,
    questions: List[Dict[str, Any]],
    top_k: int = 3,
    budget: int = 800,
    out_dir: Path = None
) -> Dict[str, Any]:
    traces = []
    latencies = []
    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0
    total_context_chars = 0

    for q in questions:
        query_text = q["question"]
        gold = q["gold"]
        filters = q.get("filter")

        t0 = time.perf_counter()
        results = retriever.search(query_text, top_k=top_k, filters=filters)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        # Hit@1 and Hit@3
        h1 = hit(results[0], gold) if len(results) >= 1 else False
        h3 = any(hit(r, gold) for r in results[:3])
        if h1:
            hit1_count += 1
        if h3:
            hit3_count += 1

        # Context evidence
        context_str, context_ids = pack_context(results, budget=budget)
        total_context_chars += len(context_str)
        context_chunks = [r for r in results if r["id"] in context_ids]
        c_hit = any(hit(c, gold) for c in context_chunks)
        if c_hit:
            context_hit_count += 1

        traces.append({
            "id": q["id"],
            "question": query_text,
            "filter": filters,
            "latency_ms": round(dt, 2),
            "hit1": h1,
            "hit3": h3,
            "context_hit": c_hit,
            "context_chars": len(context_str),
            "results": [
                {
                    "id": r["id"],
                    "source_id": r["source_id"],
                    "page": r["page"],
                    "score": round(r["score"], 4),
                    "hit": hit(r, gold)
                }
                for r in results
            ]
        })

    n = len(questions)
    summary = {
        "strategy": name,
        "n": n,
        "hit1": hit1_count,
        "hit1_rate": round(hit1_count / n, 4) if n else 0,
        "hit3": hit3_count,
        "hit3_rate": round(hit3_count / n, 4) if n else 0,
        "context_hit": context_hit_count,
        "context_evidence_rate": round(context_hit_count / n, 4) if n else 0,
        "avg_context_chars": round(total_context_chars / n, 1) if n else 0,
        "median_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "mean_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "service_cost_usd": 0.0
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary

def main():
    chunks_file = ROOT / "data" / "day6" / "chunks.json"
    questions_file = ROOT / "data" / "day7_questions.json"

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    questions = json.loads(questions_file.read_text(encoding="utf-8"))

    dev_questions = [q for q in questions if q["split"] == "dev"]
    test_questions = [q for q in questions if q["split"] == "test"]

    print(f"Loaded {len(chunks)} chunks, {len(dev_questions)} dev questions, {len(test_questions)} test questions.")

    print("\n--- Initializing Retrievers ---")
    t0 = time.time()
    lexical = LexicalTfidfRetriever(chunks)
    dense = DenseVectorRetriever(strategy_name="B_paragraph")
    hybrid = HybridRRFRetriever(dense, lexical, rrf_k=60)
    print(f"Initialized all retrievers in {time.time() - t0:.2f}s")

    configs = [
        ("Baseline (TF-IDF)", lexical, dev_questions, 3, 800, "dev", ROOT / "results" / "baseline_tfidf_dev"),
        ("Phương pháp 2: Dense (pgvector)", dense, dev_questions, 3, 800, "dev", ROOT / "results" / "dense_pgvector_dev"),
        ("Phương pháp 3: Hybrid (RRF)", hybrid, dev_questions, 3, 800, "dev", ROOT / "results" / "hybrid_rrf_dev"),
        ("Cải thiện 1: Dense (top-5, budget 1600)", dense, dev_questions, 5, 1600, "dev", ROOT / "results" / "improve_dense_dev"),
        ("Cải thiện 2: Hybrid (top-5, budget 1600)", hybrid, dev_questions, 5, 1600, "dev", ROOT / "results" / "improve_hybrid_dev"),
        ("Cấu hình đã chọn (Hybrid top-5, budget 1600)", hybrid, test_questions, 5, 1600, "test", ROOT / "results" / "chosen_hybrid_test"),
        ("Baseline (TF-IDF)", lexical, test_questions, 3, 800, "test", ROOT / "results" / "baseline_tfidf_test"),
    ]

    results_table = []
    print("\n--- Running Retrieval Benchmarks ---")
    for name, ret, q_set, k, budget, split, out_d in configs:
        print(f"Running: {name} on {split} (k={k}, budget={budget})...")
        s = evaluate_retriever(name, ret, q_set, top_k=k, budget=budget, out_dir=out_d)
        results_table.append({
            "name": name,
            "split": split,
            "k": k,
            "budget": budget,
            "summary": s
        })

    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP RETRIEVAL THEO ĐÚNG MẪU BÁO CÁO REPORT_TEMPLATE.md:")
    print("="*80)
    headers = ["Cấu hình", "Phương pháp, top-k, budget", "Giả thuyết / thay đổi", "Split", "Hit@1", "Evidence Hit@3", "Context Evidence Rate", "Thời gian / chi phí"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for r in results_table:
        s = r["summary"]
        n = s["n"]
        h1 = f"{s['hit1']}/{n} ({s['hit1_rate']*100:.1f}%)"
        h3 = f"{s['hit3']}/{n} ({s['hit3_rate']*100:.1f}%)"
        ch = f"{s['context_hit']}/{n} ({s['context_evidence_rate']*100:.1f}%)"
        time_cost = f"{s['median_ms']}ms / $0"
        
        notes = "Baseline lexical"
        if "pgvector" in r["name"]:
            notes = "Tìm theo ngữ nghĩa cosine vector Day 6"
        elif "Hybrid (RRF)" in r["name"] and r["k"] == 3:
            notes = "Kết hợp dense + lexical qua RRF k=60"
        elif "Cải thiện 1" in r["name"]:
            notes = "Tăng candidate k=5 và mở rộng budget 1600 để giữ trọn evidence"
        elif "Cải thiện 2" in r["name"]:
            notes = "Kết hợp RRF + tăng budget 1600 khắc phục lexical mismatch và budget overflow"
        elif "Cấu hình đã chọn" in r["name"]:
            notes = "Chốt từ Cải thiện 2 (Hybrid k=5, budget 1600)"
        elif "Baseline" in r["name"] and r["split"] == "test":
            notes = "Đối chứng trên test"

        config_col = r["name"].split(":")[0]
        params_col = f"{r['name']}, k={r['k']}, budget={r['budget']}"
        print(f"| {config_col} | {params_col} | {notes} | {r['split']} | {h1} | {h3} | {ch} | {time_cost} |")

if __name__ == "__main__":
    main()
