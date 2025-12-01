
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from typing import List, Dict
from math import ceil
from typing import List, Tuple
import random


# ============================================================================
# Helper Function: Encode Documents in Batches
# ============================================================================
# We define a small helper function that takes a list of texts (documents) and
# uses the SentenceTransformer model to compute embeddings in batches. This
# avoids running out of memory when there are many documents.
# ----------------------------------------------------------------------------
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

# ============================================================================
# Build a Simple TripletEvaluator for Validation
# ============================================================================
# For evaluation, we use a TripletEvaluator from SentenceTransformers.
# It expects three parallel lists:
#   - anchors   (e.g., questions)
#   - positives (e.g., their matching chunks)
#   - negatives (e.g., randomly sampled non-matching chunks)
#
# Here we construct **synthetic triplets** from the eval split by:
#   - using each (question, chunk) pair as (anchor, positive)
#   - sampling a random different chunk as the negative
#
# This gives us a rough "does anchor-positive score higher than anchor-negative"
# style retrieval accuracy metric for monitoring training.
# ----------------------------------------------------------------------------

def make_fake_triplets_from_pairs(ds, max_samples: int = 500) -> Tuple[List[str], List[str], List[str]]:
    """Create synthetic triplets (anchor, positive, negative) from a pairs dataset.

    Args:
        ds: A Hugging Face Dataset with columns ['question', 'chunk'].
        max_samples: Maximum number of triplets to create (for efficiency).

    Returns:
        anchors: List of anchor strings (questions).
        positives: List of positive strings (true chunks).
        negatives: List of negative strings (random other chunks).
    """
    # Ensure we do not sample more than the dataset size
    max_samples = min(max_samples, len(ds))

    anchors: List[str] = []
    positives: List[str] = []
    negatives: List[str] = []

    # Pre-collect all chunks for negative sampling
    all_chunks: List[str] = ds["chunk"]

    for i in range(max_samples):
        anchor = ds[i]["question"]
        positive = ds[i]["chunk"]

        # Sample a random negative chunk that is different from the positive
        negative = positive
        while negative == positive:
            negative = random.choice(all_chunks)

        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)

    return anchors, positives, negatives

def get_triplet_evaluator(
    eval_dataset,
    name: str = "rag2-mnrl-dev",
) -> TripletEvaluator:
    """Create a TripletEvaluator from the eval dataset."""
    # Build triplets from the eval split for use in the TripletEvaluator
    eval_anchors, eval_positives, eval_negatives = make_fake_triplets_from_pairs(
        eval_dataset,
        max_samples=500
    )

    # Create the TripletEvaluator
    dev_evaluator = TripletEvaluator(
        anchors=eval_anchors,
        positives=eval_positives,
        negatives=eval_negatives,
        name=name,
    )
    return dev_evaluator
