# ============================================================
# services/predictor.py — Orchestration Service
# Loads corpus → builds KB → exposes predict()
# ============================================================

import os
from typing import List
from model.markov_model import (
    preprocess,
    NgramKnowledgeBase,
    MarkovInferenceEngine,
)


CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")


class PredictorService:
    """
    High-level service consumed by FastAPI routes.

    On startup it:
      1. Loads the corpus from disk.
      2. Tokenises and preprocesses the text.
      3. Builds both a bigram and trigram knowledge base.
      4. Initialises an inference engine for each order.
    """

    def __init__(self, corpus_path: str = CORPUS_PATH):
        raw = self._load_corpus(corpus_path)
        self.tokens = preprocess(raw)

        # Build two knowledge bases: bigram (n=2) and trigram (n=3)
        self.kb_bigram  = NgramKnowledgeBase().build(self.tokens, n=2)
        self.kb_trigram = NgramKnowledgeBase().build(self.tokens, n=3)

        self.engine_bigram  = MarkovInferenceEngine(self.kb_bigram,  smoothing=True)
        self.engine_trigram = MarkovInferenceEngine(self.kb_trigram, smoothing=True)

        print(f"[PredictorService] Corpus loaded: {len(self.tokens)} tokens, "
              f"{len(self.kb_bigram.vocab)} unique words")

    # ── Public API ──────────────────────────────────────────

    def predict(self, text: str, top_k: int = 5, n: int = 2) -> List[dict]:
        """
        Given user text (evidence), return top-k next-word predictions.

        Steps:
          1. Preprocess user input into tokens.
          2. Extract the context window (last n-1 tokens).
          3. Run the Markov inference engine.
          4. Return sorted predictions.
        """
        user_tokens = preprocess(text)
        if not user_tokens:
            return []

        engine = self.engine_bigram if n == 2 else self.engine_trigram
        context = engine.get_context(user_tokens)

        if context is None:
            return []

        predictions = engine.predict(context=context, top_k=top_k)
        return predictions

    def get_stats(self) -> dict:
        """Return knowledge-base statistics for the /stats endpoint."""
        return {
            "total_tokens": len(self.tokens),
            "unique_words": len(self.kb_bigram.vocab),
            "bigram_contexts": len(self.kb_bigram.ngram_counts),
            "trigram_contexts": len(self.kb_trigram.ngram_counts),
        }

    # ── Private helpers ─────────────────────────────────────

    @staticmethod
    def _load_corpus(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
