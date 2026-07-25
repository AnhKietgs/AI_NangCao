# ============================================================
# model/markov_model.py — Knowledge Base + Inference Engine
#
# ACADEMIC NOTE:
#   Knowledge Base  = the n-gram frequency table (self.ngram_counts)
#   Inference Engine = probability calculation via Markov property
# ============================================================

import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# PART 1: TEXT PREPROCESSING  (tokenisation + cleaning)
# ──────────────────────────────────────────────────────────────

# def preprocess(text: str) -> List[str]:
#     """
#     Clean raw text and return a list of lowercase tokens.

#     Steps:
#       1. Lower-case everything.
#       2. Remove non-alphabetic characters (keep spaces).
#       3. Split on whitespace.
#       4. Filter empty strings.
#     """
#     text = text.lower()
#     text = re.sub(r"[^a-z\s]", " ", text)   # keep only a-z and spaces
#     tokens = text.split()
#     return [t for t in tokens if t]          # remove empty strings

def preprocess(text: str) -> List[str]:
    # Chuyển về chữ thường
    text = text.lower()
    
    # Regex mới: Giữ lại a-z, khoảng trắng VÀ các ký tự tiếng Việt có dấu
    # Bạn có thể dùng dải ký tự cụ thể hoặc đơn giản là giữ lại chữ cái Unicode
    text = re.sub(r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text)
    
    return [t for t in text.split() if t]
# ──────────────────────────────────────────────────────────────
# PART 2: KNOWLEDGE BASE  — N-gram frequency table
# ──────────────────────────────────────────────────────────────

class NgramKnowledgeBase:
    """
    Stores the n-gram counts built from the training corpus.

    self.ngram_counts  : Dict[(context_tuple) -> Counter{next_word: count}]
    self.context_totals: Dict[(context_tuple) -> int]   total tokens after context
    self.vocab         : set of all unique words
    """

    def __init__(self):
        # ngram_counts[(w1,)] = Counter({'word_a': 42, 'word_b': 17, ...})
        self.ngram_counts: Dict[Tuple, Counter] = defaultdict(Counter)
        self.context_totals: Dict[Tuple, int] = defaultdict(int)
        self.vocab: set = set()
        self.n_order: int = 2          # set during build()
        self.total_tokens: int = 0

    def build(self, tokens: List[str], n: int = 2):
        """
        Build the n-gram table from a token list.

        For n=2 (bigram):  context=(w_i,)           next=w_{i+1}
        For n=3 (trigram): context=(w_i, w_{i+1})   next=w_{i+2}

        Example (bigram, tokens=['the','cat','sat']):
          ngram_counts[('the',)]['cat'] += 1
          ngram_counts[('cat',)]['sat'] += 1
        """
        self.n_order = n
        self.vocab.update(tokens)
        self.total_tokens = len(tokens)
        context_size = n - 1            # bigram=1, trigram=2

        for i in range(len(tokens) - context_size):
            context = tuple(tokens[i: i + context_size])   # e.g. ('the',) or ('the','cat')
            next_word = tokens[i + context_size]
            self.ngram_counts[context][next_word] += 1
            self.context_totals[context] += 1

        return self   # fluent interface


# ──────────────────────────────────────────────────────────────
# PART 3: INFERENCE ENGINE  — Markov transition probabilities
# ──────────────────────────────────────────────────────────────

class MarkovInferenceEngine:
    """
    Computes P(next_word | context) using the Markov property:

        P(w_n | w_1 ... w_{n-1}) ≈ P(w_n | w_{n-k+1} ... w_{n-1})

    where k is the n-gram order (2 or 3).

    Laplace (add-1) smoothing is applied to handle unseen n-grams:

        P_smooth(w | context) = (count(context, w) + 1)
                                ─────────────────────────
                                (total(context) + |V|)

    where |V| is the vocabulary size.
    """

    def __init__(self, kb: NgramKnowledgeBase, smoothing: bool = True):
        self.kb = kb
        self.smoothing = smoothing
        self.V = len(kb.vocab)          # vocabulary size

    def get_context(self, tokens: List[str]) -> Optional[Tuple]:
        """
        Extract the relevant context window from user input tokens.
        For bigram: last 1 token.  For trigram: last 2 tokens.
        Returns None if there are not enough tokens.
        """
        ctx_size = self.kb.n_order - 1
        if len(tokens) < ctx_size:
            return None
        return tuple(tokens[-ctx_size:])

    def predict(self, context: Tuple, top_k: int = 5) -> List[dict]:
        """
        Main inference method.

        Given a context tuple, return the top-k next words sorted
        by descending probability.

        Returns: [{"word": str, "probability": float, "count": int}, ...]
        """
        counts = self.kb.ngram_counts.get(context, Counter())
        total  = self.kb.context_totals.get(context, 0)

        if not counts and not self.smoothing:
            return []

        # ── Rank candidates ──────────────────────────────────
        # If context was seen: rank by real counts + smoothing
        # If context unseen  : fall back to unigram (word frequency)
        if counts:
            candidates = counts
            ctx_total  = total
        else:
            # fall back: flatten all next-word counts as unigram proxy
            all_counts: Counter = Counter()
            for c in self.kb.ngram_counts.values():
                all_counts.update(c)
            candidates = all_counts
            ctx_total  = sum(all_counts.values())

        results = []
        for word, count in candidates.most_common(top_k * 3):   # over-sample then re-rank
            if self.smoothing:
                prob = (count + 1) / (ctx_total + self.V)
            else:
                prob = count / ctx_total if ctx_total else 0.0
            results.append({
                "word": word,
                "probability": round(prob, 6),
                "count": count
            })

        # Sort descending by probability, return top_k
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results[:top_k]

    def log_prob(self, tokens: List[str]) -> float:
        """
        Compute log-probability of a token sequence (useful for evaluation).
        """
        ctx_size = self.kb.n_order - 1
        log_p = 0.0
        for i in range(ctx_size, len(tokens)):
            context = tuple(tokens[i - ctx_size: i])
            word    = tokens[i]
            count   = self.kb.ngram_counts[context][word]
            total   = self.kb.context_totals[context]
            if self.smoothing:
                prob = (count + 1) / (total + self.V)
            else:
                prob = count / total if total else 1e-10
            log_p += math.log(prob)
        return log_p
