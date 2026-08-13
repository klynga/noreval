import os
import sys

import datasets

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


def filter_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda example: len(example["fact"]) > 0)


def _gold_indices(doc):
    return [doc["choices"]["label"].index(doc["answer"])]


process_results = make_process_results(_gold_indices)
