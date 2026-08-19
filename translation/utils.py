"""Pooled corpus BLEU/chrF over the K sampled translations per source.

process_results emits K (reference, prediction) pairs per document; the
aggregations pool the pairs of all documents, so every source appears K times
in the corpus.  The `bleu_stderr`/`chrf_stderr` companion metrics carry the
standard errors: per-sentence sufficient statistics are extracted once, then
documents are resampled with replacement, each document's K pairs moving
together.
"""

import logging
import random
import statistics

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


logger = logging.getLogger(__name__)

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 1234


def _flatten(items):
    return [pair for doc_pairs in items for pair in doc_pairs]


def pooled_bleu(items):
    pairs = _flatten(items)
    return sacrebleu.corpus_bleu(
        [prediction for _, prediction in pairs],
        [[reference for reference, _ in pairs]],
    ).score


def pooled_chrf(items):
    pairs = _flatten(items)
    return sacrebleu.corpus_chrf(
        [prediction for _, prediction in pairs],
        [[reference for reference, _ in pairs]],
    ).score


def bleu_stderr(items):
    return _bootstrap_stderr(items, BLEU(), pooled_bleu(items))


def chrf_stderr(items):
    return _bootstrap_stderr(items, CHRF(), pooled_chrf(items))


def _bootstrap_stderr(items, metric, direct_score):
    """Bootstrap the corpus score by resampling documents.

    Uses sacrebleu's per-sentence statistics, verified against the directly
    computed corpus score, so each replicate is a cheap re-aggregation."""
    try:
        pairs = _flatten(items)
        rows = metric._extract_corpus_statistics(
            [prediction for _, prediction in pairs],
            [[reference for reference, _ in pairs]],
        )
        if abs(metric._aggregate_and_compute(rows).score - direct_score) > 1e-6:
            raise ValueError("statistics do not reproduce the corpus score")

        doc_rows, offset = [], 0
        for doc_pairs in items:
            doc_rows.append(rows[offset : offset + len(doc_pairs)])
            offset += len(doc_pairs)

        rng = random.Random(BOOTSTRAP_SEED)
        n = len(doc_rows)
        replicates = []
        for _ in range(BOOTSTRAP_ITERS):
            sample = [row for _ in range(n) for row in doc_rows[rng.randrange(n)]]
            replicates.append(metric._aggregate_and_compute(sample).score)
        return statistics.stdev(replicates)
    except Exception as exc:
        logger.error("bootstrap stderr failed (%s): %s", type(metric).__name__, exc)
        return float("nan")
