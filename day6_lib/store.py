import json
import os
from typing import Any, Dict, List, Optional

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from day6_lib.models import DocumentChunk, SearchResult

load_dotenv()


class PostgresVectorStore:

    DIMENSION = 384

    def __init__(self):
        self.db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:55433/day6_lab",
        )
        self._init_db()

    def _init_db(self):
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding vector({self.DIMENSION}) NOT NULL
                    );
                """)
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS chunks_metadata_gin ON chunks USING GIN (metadata);"
                )
            conn.commit()

    def count(self, strategy_name: Optional[str] = None) -> int:
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                if strategy_name:
                    cur.execute(
                        "SELECT COUNT(*) FROM chunks WHERE metadata->>'strategy' = %s",
                        (strategy_name,),
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM chunks")
                return cur.fetchone()[0]

    def insert_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return

        strategies = {c.metadata.get("strategy") for c in chunks}
        documents = {c.document_id for c in chunks}

        if len(strategies) != 1 or None in strategies:
            raise ValueError("Chunks phải thuộc đúng một strategy.")

        strategy = next(iter(strategies))

        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:

                for document_id in documents:
                    cur.execute(
                        """
                        DELETE FROM chunks
                        WHERE document_id = %s AND metadata->>'strategy' = %s
                        """,
                        (document_id, strategy),
                    )

                for chunk in chunks:
                    if chunk.embedding is None:
                        raise ValueError(f"Missing embedding: {chunk.chunk_id}")
                    if len(chunk.embedding) != self.DIMENSION:
                        raise ValueError("Embedding dimension không hợp lệ.")

                    embedding_str = "[" + ",".join(str(float(x)) for x in chunk.embedding) + "]"

                    cur.execute(
                        """
                        INSERT INTO chunks (chunk_id, document_id, source, page, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chunk.text,
                            json.dumps(chunk.metadata),
                            embedding_str,
                        ),
                    )
            conn.commit()

        print(f"Stored {len(chunks)} chunks: {strategy}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        strategy_name: str = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:

        if not strategy_name:
            raise ValueError("strategy_name là bắt buộc.")
        if len(query_embedding) != self.DIMENSION:
            raise ValueError("Query embedding dimension không hợp lệ.")

        embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

        filters = {"strategy": strategy_name}
        filters.update(filter_dict or {})

        sql = """
            SELECT chunk_id, document_id, text, source, page, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE metadata @> %s::jsonb
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (embedding_str, json.dumps(filters), embedding_str, top_k)

        results = []
        with psycopg.connect(self.db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    results.append(SearchResult(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        text=row["text"],
                        source=row["source"],
                        page=row["page"],
                        score=float(row["score"]),
                        metadata=row["metadata"],
                    ))
        return results
