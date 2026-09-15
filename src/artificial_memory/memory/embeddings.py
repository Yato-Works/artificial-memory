from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=8)
def get_sentence_transformer(model_name: str):
    """Load (and cache) a SentenceTransformer model.

    Models are expensive to load (multi-GB torch import + model weights),
    so all engines share a single cached instance per model name.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
