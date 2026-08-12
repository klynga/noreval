import os
import sys

# noreval_metrics.py lives at the repository root; the harness imports each
# task's utils.py standalone, so locate the root by walking up from here
_root = os.path.dirname(os.path.abspath(__file__))
while not os.path.isfile(os.path.join(_root, "noreval_metrics.py")):
    _parent = os.path.dirname(_root)
    assert _parent != _root, "noreval_metrics.py not found in any parent directory"
    _root = _parent
if _root not in sys.path:
    sys.path.insert(0, _root)

from noreval_metrics import variant_process_results

_here = os.path.dirname(os.path.abspath(__file__))


def _gold_indices(doc):
    return [doc["sentiment"]]


def _add_f1_pairs(metrics, predictions, doc):
    for suffix, pred in predictions.items():
        metrics[f"f1{suffix}"] = (doc["sentiment"], pred)


def __getattr__(name):
    # utils.process_<variant> scores the doc_to_choice of <variant>.yaml,
    # so new prompt variants need no code changes here
    return variant_process_results(_here, name, _gold_indices, _add_f1_pairs)
