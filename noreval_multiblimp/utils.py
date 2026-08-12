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

from noreval_metrics import score_choices


def process_results(doc, results):
    metrics, _ = score_choices(results, [doc["sen"], doc["wrong_sen"]], [0])
    return metrics


def filter_dataset_1_2(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "1 -> 2")

def filter_dataset_2_1(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "2 -> 1")

def filter_dataset_3_1(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "3 -> 1")

def filter_dataset_1_3(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "1 -> 3")

def filter_dataset_3_2(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "3 -> 2")

def filter_dataset_2_3(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "2 -> 3")

def filter_dataset_1_23(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "1 -> 2|3")

def filter_dataset_sg_du(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "SG -> DU")

def filter_dataset_du_sg(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "DU -> SG")

def filter_dataset_sg_pl(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "SG -> PL")

def filter_dataset_pl_sg(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "PL -> SG")

def filter_dataset_pl_du(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "PL -> DU")

def filter_dataset_du_pl(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "DU -> PL")

