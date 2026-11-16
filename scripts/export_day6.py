import json, os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
url = os.getenv("DATABASE_URL")

with psycopg.connect(url) as conn, conn.cursor() as cur:
    cur.execute(
        "SELECT chunk_id, document_id, source, page, text, metadata "
        "FROM chunks WHERE metadata->>'strategy' = %s ORDER BY page, chunk_id",
        ("B_paragraph",),
    )
    chunks = [{"id": r[0], "text": r[4], "source_id": r[1], "source": r[2],
               "page": r[3], "metadata": r[5]} for r in cur.fetchall()]

out_dir = ROOT / "data" / "day6"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "chunks.json"
out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Exported {len(chunks)} chunks to {out_file}")
