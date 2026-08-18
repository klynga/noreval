"""Shared scoring helpers for the NorEval multiple-choice tasks.

lm-eval's built-in multiple_choice pipeline computes acc, acc_norm and
acc_mutual_info, but has no continuous "probability of the correct answer"
metrics. Tasks that want them define a custom process_results, which replaces
the built-in computation entirely, so the accuracy metrics are re-implemented
here as well (with the same formulas as the harness).

Each task passes its choices function to `make_process_results` and points the
yaml's `doc_to_choice` at the *same* function, so the strings whose
loglikelihoods the harness scores are, by construction, the strings used for
the per-character normalization here — without replicating them and without
requiring a patched harness.

The metrics come in three normalization variants, mirroring the harness:
  * ""             raw conditional loglikelihoods,
  * "_norm"        loglikelihoods divided by the choice length in characters
                   (a no-op whenever all choices have equal length, e.g. the
                   single-letter options of the mcf formulations), and
  * "_mutual_info" pointwise mutual information: conditional minus
                   unconditional loglikelihoods.  For tasks whose prompt is
                   empty ("" doc_to_text), the two coincide in zero-shot runs,
                   every score ties at 0, and the argmax degenerates to choice
                   0 — technically correct, but only informative with fewshot
                   context.

For each variant, three metrics are produced:
  * "acc*":          the argmax prediction is one of the correct choices,
  * "prob_correct*": the probability mass that the softmax over the per-choice
                     scores puts on the correct choice(s) — the continuous
                     analogue of the accuracy.  (For "_norm" and
                     "_mutual_info" this is a softmax over rescaled
                     loglikelihoods, not a calibrated probability.), and
  * "loglikelihood_correct*": the per-variant score of the correct choice
                     itself, passed through without comparing it to the other
                     choices (the best correct choice, if several are gold).

This module lives at the repository root so that every task can share it. The
harness imports each task's utils.py standalone, so the task modules load this
file after a  sys.path.insert(0, <repository root>)  one-liner; the consistent
task/language/ file structure puts the root at "../.." from every utils.py
that needs it.
"""

from scipy.special import softmax


def make_process_results(choices, gold_indices, postprocess=None, expect_mutual_info=True):
    """Build a task's process_results from its choice and gold definitions.

    choices(doc) returns the rendered answer choices and must be the very
    function referenced by the task's doc_to_choice; gold_indices(doc) returns
    the indices of the doc's correct choices (usually just one).  The optional
    postprocess(metrics, predictions, doc) can add task-specific entries such
    as the (prediction, gold) pairs of F1.  expect_mutual_info declares
    whether the task's metric_list requests the "_mutual_info" variants; see
    split_loglikelihoods.
    """

    def process_results(doc, results):
        metrics, predictions = score_choices(
            results, choices(doc), gold_indices(doc), expect_mutual_info
        )
        if postprocess is not None:
            postprocess(metrics, predictions, doc)
        return metrics

    return process_results


def split_loglikelihoods(results, n_choices, expect_mutual_info):
    """Split the flat result list into conditional and unconditional loglikelihoods.

    The harness sends one loglikelihood request per answer choice.  When the
    literal metric name "acc_mutual_info" is in the metric list, it appends one
    extra unconditional ("", choice) request per choice after the conditional
    ones.  It does not tell process_results which case occurred, so the
    doubled length is the only signal that the unconditional loglikelihoods
    (the PMI denominators) are present.

    Because the harness keys that decision on "acc_mutual_info" alone, a
    metric list with some other "*_mutual_info" metric but without
    acc_mutual_info would silently get no unconditional requests; with
    expect_mutual_info=True (the default) that misconfiguration raises here
    instead of surfacing as a missing-metric error at aggregation time.
    """
    lls = [result[0] if isinstance(result, (tuple, list)) else result for result in results]
    if len(lls) == 2 * n_choices:
        return lls[:n_choices], lls[n_choices:]
    if len(lls) == n_choices:
        if expect_mutual_info:
            raise ValueError(
                "no unconditional loglikelihoods received although a "
                "'*_mutual_info' metric is expected: the harness only sends "
                "them when the literal name 'acc_mutual_info' is in "
                "metric_list — add it (or pass expect_mutual_info=False)"
            )
        return lls, []
    raise ValueError(f"Expected {n_choices} or {2 * n_choices} loglikelihoods, got {len(lls)}")


def score_choices(results, choices, gold_indices, expect_mutual_info=True):
    """Score one document of a multiple-choice task.

    Returns (metrics, predictions); predictions maps the same variant suffixes
    to the predicted choice index, for tasks that additionally forward
    (prediction, gold) pairs to metrics such as F1.
    """
    conditional_lls, unconditional_lls = split_loglikelihoods(
        results, len(choices), expect_mutual_info
    )

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
        metrics[f"loglikelihood_correct{suffix}"] = float(max(lls[i] for i in gold_indices))
    return metrics, predictions
