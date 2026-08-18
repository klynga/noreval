import random
import statistics

import numpy as np
import sklearn

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 1234


def multi_f1(items):
    """
    Computes the macro-average F1 score.

    zero_division=0 makes degenerate bootstrap replicates (all predictions in
    one class) silent; sklearn's default computes the same value but warns.
    """
    preds, golds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    fscore = sklearn.metrics.f1_score(golds, preds, average="macro", zero_division=0)
    return fscore


def multi_f1_stderr(items):
    """Question-level bootstrap standard error of the macro-F1.

    The harness has no closed form for corpus-level estimators
    (arXiv:2411.00640), so the (prediction, gold) pairs are resampled with
    replacement and `multi_f1` itself is recomputed on every replicate -- the
    stderr describes exactly the estimator reported as the point estimate.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    pairs = list(items)
    n = len(pairs)
    replicates = [
        multi_f1([pairs[rng.randrange(n)] for _ in range(n)])
        for _ in range(BOOTSTRAP_ITERS)
    ]
    return statistics.stdev(replicates)
