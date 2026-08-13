import os
import sys

import numpy as np
import sklearn

# noreval_metrics.py lives at the repository root; the harness imports each
# task's utils.py standalone, so locate the root by walking up from here
_root = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_root, "noreval_metrics.py")):
    _parent = os.path.dirname(_root)
    assert _parent != _root, "noreval_metrics.py not found in any parent directory"
    _root = _parent
if _root not in sys.path:
    sys.path.insert(0, _root)

from noreval_metrics import make_process_results


def multi_f1(items):
    """
    Computes the macro-average F1 score.
    """
    golds, preds = zip(*items)
    golds = np.array(golds)
    preds = np.array(preds)
    fscore = sklearn.metrics.f1_score(golds, preds, average="macro")
    return fscore


def _add_f1_pairs(metrics, predictions, doc):
    for suffix, pred in predictions.items():
        metrics[f"f1{suffix}"] = (doc["sentiment"], pred)


process_results = make_process_results(lambda doc: [doc["sentiment"]], _add_f1_pairs)
