"""Shared scoring helpers for the NorEval multiple-choice tasks.

lm-eval's built-in multiple_choice pipeline computes acc, acc_norm and
acc_mutual_info, but has no continuous "probability of the correct answer"
metrics. Tasks that want them define a custom process_results, which replaces
the built-in computation entirely, so the accuracy metrics are re-implemented
here as well (with the same formulas as the harness).

The process_results functions built here declare a `choices` parameter, which
our lm-evaluation-harness fork detects and fills with the rendered
doc_to_choice of the scored document — the exact strings whose loglikelihoods
arrive in `results` — so tasks never have to replicate their choice strings.

This module lives at the repository root so that every task can share it. The
harness imports each task's utils.py standalone, so the task modules load this
file by walking up the directory tree; see e.g. ncb/utils.py.
"""

from scipy.special import softmax


def make_process_results(gold_indices, postprocess=None):
    """Build a task's process_results from its notion of the correct answers.

    gold_indices(doc) returns the indices of the doc's correct choices (usually
    just one); the optional postprocess(metrics, predictions, doc) can add
    task-specific entries such as the (gold, prediction) pairs of F1.
    """

    def process_results(doc, results, choices):
        metrics, predictions = score_choices(results, choices, gold_indices(doc))
        if postprocess is not None:
            postprocess(metrics, predictions, doc)
        return metrics

    return process_results


def split_loglikelihoods(results, n_choices):
    """Split the flat result list into conditional and unconditional loglikelihoods.

    The harness sends one loglikelihood request per answer choice. When
    acc_mutual_info (or prob_correct_mutual_info) is in the metric list, it
    appends one extra unconditional ("", choice) request per choice after the
    conditional ones. It does not tell process_results which case occurred, so
    the doubled length is the only signal that the unconditional loglikelihoods
    (the PMI denominators) are present.
    """
    lls = [result[0] if isinstance(result, (tuple, list)) else result for result in results]
    if len(lls) not in (n_choices, 2 * n_choices):
        raise ValueError(
            f"Expected {n_choices} or {2 * n_choices} loglikelihoods, got {len(lls)}"
        )
    return lls[:n_choices], lls[n_choices:]


def score_choices(results, choices, gold_indices):
    """Score one document of a multiple-choice task.

    For the raw loglikelihoods and for each normalization (per character, and
    PMI whenever the harness sent the unconditional requests), computes:
      * the accuracy of the argmax prediction
        ("acc", "acc_norm", "acc_mutual_info"), and
      * the probability mass that the softmax over the per-choice scores puts
        on the correct choice(s) ("prob_correct", "prob_correct_norm",
        "prob_correct_mutual_info"), the continuous analogue of the accuracy.

    gold_indices lists the indices of all correct choices (usually just one).

    Returns (metrics, predictions); predictions maps the same suffixes to the
    predicted choice index, for tasks that additionally forward (gold, pred)
    pairs to metrics such as F1.
    """
    conditional_lls, unconditional_lls = split_loglikelihoods(results, len(choices))

    scored_lls = {
        "": conditional_lls,
        "_norm": [ll / len(choice) for ll, choice in zip(conditional_lls, choices)],
    }
    if unconditional_lls:
        scored_lls["_mutual_info"] = [
            conditional - unconditional
            for conditional, unconditional in zip(conditional_lls, unconditional_lls)
        ]

    metrics, predictions = {}, {}
    for suffix, lls in scored_lls.items():
        pred = max(range(len(choices)), key=lambda i: lls[i])
        probs = softmax(lls)
        predictions[suffix] = pred
        metrics[f"acc{suffix}"] = int(pred in gold_indices)
        metrics[f"prob_correct{suffix}"] = float(sum(probs[i] for i in gold_indices))
    return metrics, predictions
