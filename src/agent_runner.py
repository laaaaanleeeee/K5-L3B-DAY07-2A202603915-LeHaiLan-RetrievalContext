"""Agent benchmark runner — uses a real LLM controller when an API key is set.

Run:
    python src/agent_runner.py

Required:
    Set GEMINI_API_KEY (or OPENAI_API_KEY / OPENROUTER_API_KEY) in .env to
    use a real model. Without a key, falls back to rule-based controller and
    prints a warning.
"""
import json, os, re, sys, time
from pathlib import Pathfrom typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT / "starter"))

from src.retrievers import (
    DenseVectorRetriever,
    LexicalTfidfRetriever,
    HybridRRFRetriever,
    pack_context,
)
from src.llm_controller import make_controller
from starter.agentic import run_agent


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").casefold())


class AgentRetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.retriever.search(query, top_k=top_k, filters=filters)


def run_agent_benchmark(
    split: str,
    retriever,
    chunk_map: Dict[str, Dict[str, Any]],
    top_k: int = 5,
    budget: int = 1600,
    max_searches: int = 2,
    output_dir: Path = None,
    llm_prefer: str = "auto",
    inter_query_delay: float = 0.0,
) -> Dict[str, Any]:
    benchmark_file = ROOT / "data" / "day7_benchmark.json"
    benchmark_all = json.loads(benchmark_file.read_text(encoding="utf-8"))

    questions = [q for q in benchmark_all if q["split"] == split]
    q_map = {q["id"]: q for q in questions}

    controller = make_controller(q_map, chunk_map, max_searches=max_searches, prefer=llm_prefer)
    retriever_wrapper = AgentRetrieverWrapper(retriever)

    traces = []
    reviews = []
    cases = []

    for q in questions:
        qid = q["id"]
        q_text = q["question"]
        is_answerable = q.get("answerable", True)

        cases.append({"id": qid, "split": split, "answerable": is_answerable})

        try:
            res = run_agent(
                question=q_text,
                controller=controller,
                retriever=retriever_wrapper,
                filters=q.get("filter"),
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            status = res["status"]
        except Exception as exc:
            print(f"  [ERROR] {qid}: {exc}")
            res = {"status": "error", "answer": None, "citations": [], "events": []}
            status = "error"

        if inter_query_delay > 0:
            time.sleep(inter_query_delay)

        trace_entry = {
            "query_id": qid,
            "question": q_text,
            "status": status,
            "answer": res.get("answer"),
            "citations": res.get("citations", []),
            "events": res.get("events", []),
        }
        traces.append(trace_entry)

        citations = res.get("citations", [])
        citation_support = []

        if status != "error":
            for c in citations:
                if is_answerable:
                    gold_ev = [norm(g["evidence"]) for g in q.get("gold", [])]
                    all_context = "".join(
                        e.get("context", "") for e in res.get("events", [])
                    )
                    supported = any(gev in norm(all_context) for gev in gold_ev)
                    citation_support.append(supported)
                else:
                    citation_support.append(False)

        review_entry: Dict[str, Any] = {
            "query_id": qid,
            "citation_support": citation_support,
        }

        if status == "error":
            review_entry["answer_grade"] = 0
            review_entry["error"] = True
        elif is_answerable:
            if status == "answered" and all(citation_support) and citation_support:
                review_entry["answer_grade"] = 2
            elif status == "answered":
                review_entry["answer_grade"] = 1
            else:
                review_entry["answer_grade"] = 0
        else:
            review_entry["abstention_correct"] = status == "abstain"

        reviews.append(review_entry)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cases.json").write_text(
            json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "reviews.json").write_text(
            json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return {"cases": cases, "traces": traces, "reviews": reviews}


def main():
    chunks_file = ROOT / "data" / "day6" / "chunks.json"
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunk_map = {c["id"]: c for c in chunks}

    print("--- Initializing Agent Retrievers ---")
    lexical = LexicalTfidfRetriever(chunks)
    dense = DenseVectorRetriever(strategy_name="B_paragraph")
    hybrid = HybridRRFRetriever(dense, lexical, rrf_k=60)

    configs = [
        (
            "Agent Baseline Dev (TF-IDF, k=3, budget=800)",
            lexical, "dev", 3, 800,
            ROOT / "results" / "agent_baseline_dev",
        ),
        (
            "Agent Chosen Dev (Hybrid, k=5, budget=1600)",
            hybrid, "dev", 5, 1600,
            ROOT / "results" / "agent_chosen_dev",
        ),
        (
            "Agent Baseline Test (TF-IDF, k=3, budget=800)",
            lexical, "test", 3, 800,
            ROOT / "results" / "agent_baseline_test",
        ),
        (
            "Agent Chosen Test (Hybrid, k=5, budget=1600)",
            hybrid, "test", 5, 1600,
            ROOT / "results" / "agent_chosen_test",
        ),
        (
            "Agent Improve3 Dev (Hybrid, k=5, budget=2400, Gemini)",
            hybrid, "dev", 5, 2400,
            ROOT / "results" / "agent_improve3_dev",
        ),
        (
            "Agent Improve3 Test (Hybrid, k=5, budget=2400, Gemini)",
            hybrid, "test", 5, 2400,
            ROOT / "results" / "agent_improve3_test",
        ),
    ]

    from starter.agent_metrics import aggregate

    for label, ret, split, k, budget, out_dir in configs:
        print(f"\nRunning: {label}")
        result = run_agent_benchmark(
            split=split,
            retriever=ret,
            chunk_map=chunk_map,
            top_k=k,
            budget=budget,
            max_searches=2,
            output_dir=out_dir,
            llm_prefer="auto",
        )
        cases_data = json.loads((out_dir / "cases.json").read_text(encoding="utf-8"))
        traces_data = json.loads((out_dir / "traces.json").read_text(encoding="utf-8"))
        reviews_data = json.loads((out_dir / "reviews.json").read_text(encoding="utf-8"))
        summary = aggregate(cases_data, traces_data, reviews_data)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{label}]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
