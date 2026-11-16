from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    source: str
    page: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    source: str
    page: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)