import json, math, os, re, unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())

def pack_context(chunks: List[Dict[str, Any]], budget: int = 800) -> Tuple[str, List[str]]:
    """Packs chunks into context within budget limit.

    Strategy: fit whole chunks in rank order; if the last chunk overflows,
    truncate it at a word boundary rather than dropping it entirely, so that
    partial evidence is not silently lost.
    """
    lines, ids, used = [], [], 0
    for c in chunks:
        cid = c["id"]
        if cid in ids:
            continue
        line = f"[{cid}] (nguồn: {c.get('source_id', c.get('source'))}, trang {c.get('page')}) {c.get('text', '')}".strip()
        sep = 1 if lines else 0
        cost = len(line) + sep
        if used + cost <= budget:
            lines.append(line)
            ids.append(cid)
            used += cost
        else:
            remaining = budget - used - sep
            if remaining >= 80:
                truncated = line[:remaining].rsplit(" ", 1)[0]
                if truncated:
                    lines.append(truncated)
                    ids.append(cid)
            break
    return "\n".join(lines), ids


class LexicalTfidfRetriever:
    """Offline Lexical TF-IDF Retriever matching starter/baseline.py."""
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.chunk_by_id = {c["id"]: c for c in chunks}
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_counts = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.N = max(1, len(chunks))
        
        df = Counter()
        for c in self.doc_counts:
            df.update(c.keys())
        self.idf = {t: math.log((self.N + 1) / (count + 0.5)) + 1.0 for t, count in df.items()}

    def search(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        q_counts = Counter(q_tokens)
        if not q_tokens:
            candidates = self.chunks[:top_k]
            return [{**c, "score": 0.0} for c in candidates]

        scores = []
        for i, (counts, length) in enumerate(zip(self.doc_counts, self.doc_lens)):
            chunk = self.chunks[i]
            if filters:
                match = True
                for k, v in filters.items():
                    if chunk.get("metadata", {}).get(k) != v and chunk.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            score = 0.0
            if length > 0:
                for term, qf in q_counts.items():
                    tf = counts.get(term, 0)
                    if tf > 0:
                        score += (tf / length) * self.idf.get(term, 0.0) * qf
            scores.append((score, i))

        scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [{**self.chunks[i], "score": float(s)} for s, i in scores[:top_k]]


class DenseVectorRetriever:
    """Dense retriever using Day 6 PostgresVectorStore and all-MiniLM-L6-v2."""
    def __init__(self, strategy_name: str = "B_paragraph"):
        from day6_lib.embeddings import EmbeddingModel
        from day6_lib.store import PostgresVectorStore
        self.model = EmbeddingModel("all-MiniLM-L6-v2")
        self.store = PostgresVectorStore()
        self.strategy_name = strategy_name

    def search(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        q_vec = self.model.encode_text(query)
        results = self.store.search(q_vec, top_k=top_k, strategy_name=self.strategy_name, filter_dict=filters)
        return [
            {
                "id": r.chunk_id,
                "text": r.text,
                "source_id": r.document_id,
                "source": r.source,
                "page": r.page,
                "score": float(r.score),
                "metadata": r.metadata
            }
            for r in results
        ]


class HybridRRFRetriever:
    """Hybrid Retriever combining Dense and Lexical results using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseVectorRetriever, lexical_retriever: LexicalTfidfRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None, candidate_k: int = 15) -> List[Dict[str, Any]]:
        dense_results = self.dense.search(query, top_k=candidate_k, filters=filters)
        lexical_results = self.lexical.search(query, top_k=candidate_k, filters=filters)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, r in enumerate(dense_results, 1):
            cid = r["id"]
            chunk_map[cid] = r
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, r in enumerate(lexical_results, 1):
            cid = r["id"]
            if cid not in chunk_map:
                chunk_map[cid] = r
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        return [{**chunk_map[cid], "score": float(rrf_scores[cid])} for cid in sorted_cids]

