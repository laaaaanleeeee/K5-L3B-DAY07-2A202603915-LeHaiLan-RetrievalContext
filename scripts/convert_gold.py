import json, os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def find_exact_slice(quote: str, text: str) -> str:
    """Find the exact slice in chunk text matching quote, handling whitespace discrepancies."""
    if quote in text:
        return quote
    q_chars = [c.casefold() for c in quote if c.isalnum()]
    t_chars = [(i, c.casefold()) for i, c in enumerate(text) if c.isalnum()]
    L = len(q_chars)
    for j in range(len(t_chars) - L + 1):
        if [c for _, c in t_chars[j:j+L]] == q_chars:
            return text[t_chars[j][0] : t_chars[j+L-1][0] + 1]
    return quote

def norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", (t or "").casefold())

def main():
    chunks_path = ROOT / "data" / "day6" / "chunks.json"
    if not chunks_path.exists():
        chunks_path = ROOT / "scripts" / "data" / "day6" / "chunks.json"
    
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    old_bench = json.loads((ROOT / "scripts" / "data" / "day6_benchmark.json").read_text(encoding="utf-8"))

    answerable_questions = []
    benchmark_all = []
    cases = []
    problems = []

    for q in old_bench:
        e_norm = norm(q["evidence_quote"])
        matched_chunks = [
            c for c in chunks
            if c["source_id"] == q.get("gold_source_id")
            and c["page"] == q.get("gold_page")
            and e_norm in norm(c["text"])
        ]
        if not matched_chunks:
            matched_chunks = [
                c for c in chunks
                if c["source_id"] == q.get("gold_source_id")
                and e_norm in norm(c["text"])
            ]
        
        if not matched_chunks:
            problems.append(q["id"])

        gold_items = []
        for c in matched_chunks:
            exact_sub = find_exact_slice(q["evidence_quote"], c["text"])
            gold_items.append({
                "source_id": c["source_id"],
                "page": c["page"],
                "evidence": exact_sub
            })

        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": gold_items,
            "answerable": True,
            "expected_behavior": "search_and_answer"
        }
        if "filter" in q:
            item["filter"] = q["filter"]

        answerable_questions.append({
            "id": item["id"],
            "split": item["split"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "gold": item["gold"],
            **({"filter": item["filter"]} if "filter" in item else {})
        })
        benchmark_all.append(item)
        cases.append({
            "id": item["id"],
            "split": item["split"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })

    unanswerable = [
        {
            "id": "unans_dev_01",
            "split": "dev",
            "question": "What is the detailed hardware architecture and FLOPs requirement of the GPT-5 cluster mentioned in this textbook?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "The textbook 2501.09223v4 was written prior to any GPT-5 release and does not discuss GPT-5 cluster architecture or hardware FLOPs."
        },
        {
            "id": "unans_dev_02",
            "split": "dev",
            "question": "According to the textbook, how is Apple Neural Engine (ANE) configured specifically for running quantized Mistral-7B models?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "The document does not cover proprietary Apple Neural Engine (ANE) hardware deployment configurations for Mistral-7B."
        },
        {
            "id": "unans_test_01",
            "split": "test",
            "question": "What was the exact financial training cost in USD for the Llama-3-405B model according to chapter 3?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "The book does not provide exact proprietary financial dollar training cost records for Llama-3-405B."
        },
        {
            "id": "unans_test_02",
            "split": "test",
            "question": "How does the document describe integrating blockchain smart contracts with LLM agentic memory stores?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "The document focuses on foundational LLM architectures, pre-training, and alignment; it contains no discussion of blockchain or smart contracts."
        }
    ]

    for u in unanswerable:
        benchmark_all.append(u)
        cases.append({
            "id": u["id"],
            "split": u["split"],
            "question": u["question"],
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": u["absence_reason"]
        })

    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    
    (ROOT / "data" / "day7_questions.json").write_text(
        json.dumps(answerable_questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "data" / "day7_benchmark.json").write_text(
        json.dumps(benchmark_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "data" / "day7_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Generated:")
    print(f" - data/day7_questions.json: {len(answerable_questions)} answerable questions")
    print(f" - data/day7_benchmark.json: {len(benchmark_all)} total questions (20 answerable + {len(unanswerable)} unanswerable)")
    print(f" - data/day7_cases.json: {len(cases)} cases")
    if problems:
        print(f"Warnings - questions without evidence: {problems}")
    else:
        print("All 20 answerable questions have valid exact matching evidence in chunks!")

if __name__ == "__main__":
    main()
