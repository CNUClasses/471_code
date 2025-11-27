
# NOTE:  THIS IS THE CODE FROM RAG_Lab2_Retrieve_Rerank_HFLLM.ipynb converted to an API module 

"""
rag_pipeline_api.py

Two simple, classroom-ready APIs for a RAG pipeline:

1) load_chroma(persist_dir, collection_name, ...)
   - Loads a Chroma collection (already populated with chunked documents)
   - Loads a dense embedding model (SentenceTransformers)
   - Loads a cross-encoder reranker (SentenceTransformers CrossEncoder)
   - Returns a lightweight handle with everything needed for querying

2) query_reranked(handle, query, top_k_candidates=50, top_k_return=10, ...)
   - Step 4: Dense retrieve from Chroma using the query embedding (top_k_candidates)
   - Step 5: Prepare (query, doc) pairs
   - Step 6: Rerank top-N candidates with a cross-encoder
   - Step 7: Return the reranked chunks (id, text, metadata, scores)

This module intentionally **does not** perform generation; it returns the reranked
chunks so you can plug them into your own prompt construction & LLM of choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# 3rd party deps (pip install: chromadb, sentence-transformers, torch)
import chromadb
from chromadb.api.types import Where, WhereDocument
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class RAGHandle:
    """Lightweight handle containing everything needed to query."""
    client: Any
    collection: Any
    embedder: SentenceTransformer
    reranker: CrossEncoder
    embed_dim: int
    device: Optional[str] = None


def load_chroma(
    persist_dir: str,
    collection_name: str,
    embedding_model: str = "sentence-transformers/msmarco-distilbert-cos-v5",  #EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-cos-v5"
    
    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",  #EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-cos-v5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: Optional[str] = None,
    load_finetuned: bool = False,
) -> RAGHandle:
    """
    Load a persisted Chroma collection along with a dense embedder and a reranker.

    Args:
        persist_dir: Filesystem path where Chroma persisted the DB.
        collection_name: Name of the collection that holds your pre-chunked documents.
        embedding_model: SentenceTransformers bi-encoder for query embeddings.
        reranker_model: SentenceTransformers cross-encoder for reranking top-N candidates.
        device: Optional device override for SentenceTransformers models ("cpu", "cuda", "mps").

    Returns:
        RAGHandle with (client, collection, embedder, reranker).

    Notes:
        - The dense retriever relies on the same (or compatible) embedding model
          you used when ingesting chunks into Chroma.
        - If you used a different embedder for ingestion, retrieval quality may suffer.
    """
    # Connect to Chroma
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)

    # Load models
    if(load_finetuned==True):
        embedder= SentenceTransformer(f"models/{embedding_model}", device=device)  # bi-encoder
        print(f"Loaded finetuned embedder from models/{embedding_model}")
    else:
        embedder = SentenceTransformer(embedding_model, device=device)  # bi-encoder
        print(f"Loaded hugging face pretrained embedder: {embedding_model}")
    reranker = CrossEncoder(reranker_model, device=device)          # cross-encoder

    # Infer dimension for sanity checks if needed
    test_vec = embedder.encode(["probe"], convert_to_numpy=True)
    embed_dim = int(test_vec.shape[-1])

    return RAGHandle(
        client=client,
        collection=collection,
        embedder=embedder,
        reranker=reranker,
        embed_dim=embed_dim,
        device=device,
    )


def _to_similarity(distances: List[float]) -> List[float]:
    """
    Convert Chroma distances to a 'similarity-like' score for display only.
    Chroma returns distances (smaller is better). We map to sim = 1 / (1 + d).
    """
    return [float(1.0 / (1.0 + d)) for d in distances]


def query_reranked(
    handle: RAGHandle,
    query: str,
    top_k_candidates: int = 50,
    top_k_return: int = 10,
    where: Optional[Where] = None,
    where_document: Optional[WhereDocument] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve and rerank chunks for a query.

    Pipeline (matching Lab steps 4–7):
      4) Dense retrieve top_k_candidates from Chroma using the query embedding
      5) Build (query, candidate_doc) pairs
      6) Rerank with a CrossEncoder on the pairs
      7) Return top_k_return chunks sorted by reranker score (desc)

    Args:
        handle: RAGHandle from load_chroma().
        query: Natural-language query string.
        top_k_candidates: How many candidates to pull from Chroma before reranking.
        top_k_return: How many reranked chunks to return.
        where: (Optional) Chroma metadata filter (dictionary).
        where_document: (Optional) Chroma document filter.

    Returns:
        List of dicts with keys:
            - id: chunk id (as stored in Chroma)
            - text: chunk text
            - metadata: original metadata dict
            - dense_score: similarity-like score derived from Chroma distance
            - rerank_score: cross-encoder score (higher is better)
    """
    # 4) Dense retrieve from Chroma
    q_emb = handle.embedder.encode([query], convert_to_numpy=True)[0].tolist()
    res = handle.collection.query(
        query_embeddings=[q_emb],
        n_results=int(top_k_candidates),
        include=["documents", "metadatas", "distances", "embeddings"],
        where=where,
        where_document=where_document,
    )

    # Unpack results (Chroma returns nested lists per query)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or [0.0] * len(ids)
    dense_scores = _to_similarity(dists)

    # Early exit if nothing found
    if not ids:
        return []

    # 5) Build (query, doc) pairs
    pairs = [(query, d if isinstance(d, str) else str(d)) for d in docs]

    # 6) Cross-encoder reranking (higher = better)
    rerank_scores = handle.reranker.predict(pairs).tolist()

    # 7) Sort by rerank score desc and return top_k_return
    order = np.argsort(-np.array(rerank_scores)).tolist()

    results = []
    for idx in order[: int(top_k_return)]:
        results.append(
            {
                "id": ids[idx],
                "text": docs[idx],
                "metadata": metas[idx],
                "dense_score": float(dense_scores[idx]),
                "rerank_score": float(rerank_scores[idx]),
            }
        )
    return results


# ----------------------
# Small CLI / usage demo
# ----------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="RAG Retrieve→Rerank demo")
    parser.add_argument("--persist_dir", required=True, help="Chroma persist directory")
    parser.add_argument("--collection", required=True, help="Chroma collection name")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--top_n", type=int, default=50, help="Candidates to retrieve before rerank")
    parser.add_argument("--return_k", type=int, default=10, help="Final results to return")
    parser.add_argument("--device", default=None, help="Force device for models: cpu|cuda|mps")

    args = parser.parse_args()

    handle = load_chroma(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        device=args.device,
    )
    out = query_reranked(
        handle,
        query=args.query,
        top_k_candidates=args.top_n,
        top_k_return=args.return_k,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))