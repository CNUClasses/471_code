# ============================================================================
# Helper Function: Encode Documents in Batches
# ============================================================================
# We define a small helper function that takes a list of texts (documents) and
# uses the SentenceTransformer model to compute embeddings in batches. This
# avoids running out of memory when there are many documents.
# ----------------------------------------------------------------------------
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from math import ceil



def encode_documents(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> List[List[float]]:
    """Encode a list of documents into embeddings using the given model.

    This function:
      - Splits the input texts into batches.
      - Uses `model.encode` with `convert_to_numpy=True` and `normalize_embeddings=True`
        for cosine similarity–friendly vectors.
      - Converts the resulting numpy arrays into Python lists (for ChromaDB).

    Args:
        texts: List of string documents to encode.
        model: A SentenceTransformer model.
        batch_size: Number of documents to encode per batch.

    Returns:
        A list of embeddings, where each embedding is a list[float].
    """
    all_embeddings: List[List[float]] = []
    num_texts = len(texts)
    num_batches = ceil(num_texts / batch_size)

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_texts)
        batch_texts = texts[start:end]

        # Compute embeddings for this batch
        # - convert_to_numpy=True returns a numpy array
        # - normalize_embeddings=True L2-normalizes the embeddings
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Convert each embedding vector to a plain Python list
        for emb in batch_embeddings:
            all_embeddings.append(emb.tolist())

        # Optional logging for long runs
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"Encoded batch {i + 1}/{num_batches} (docs {start}–{end})")

    return all_embeddings
