"""Shared scoring helpers for the NorEval multiple-choice tasks.

lm-eval's built-in multiple_choice pipeline computes acc, acc_norm and
acc_mutual_info, but has no continuous "probability of the correct answer"
metrics. Tasks that want them define a custom process_results, which replaces
the built-in computation entirely, so the accuracy metrics are re-implemented
here as well (with the same formulas as the harness).

This module lives at the repository root so that every task can share it. The
harness imports each task's utils.py standalone, so the task modules load this
file by walking up the directory tree; see e.g. ncb/utils.py.
"""

import math
import os

import yaml


class _ConfigLoader(yaml.SafeLoader):
    """Reads task configs while ignoring lm-eval's custom !function tags."""


_ConfigLoader.add_constructor(
    "!function", lambda loader, node: loader.construct_scalar(node)
)


def resolve_choices(doc_to_choice, doc):
    """Turn a config's doc_to_choice into the list of scored strings for one doc,
    mirroring the harness's handling for the forms NorEval uses: either a static
    list, or a Jinja template that renders to a list literal."""
    if isinstance(doc_to_choice, list):
        return doc_to_choice
    import ast

    import jinja2

    return ast.literal_eval(jinja2.Template(doc_to_choice).render(**doc))


def variant_process_results(directory, attribute_name, gold_indices, postprocess=None):
    """Build the process_results function for one prompt variant.

    `attribute_name` must be "process_<variant>", and <variant>.yaml in
    `directory` provides the doc_to_choice whose strings are scored. Intended to
    be returned from a module-level __getattr__, so a variant config wires itself
    with `process_results: !function utils.process_<its own file stem>` and new
    variants — with any choices — need no code changes.

    This indirection exists because the harness calls a custom process_results
    with only (doc, results): it never passes the config, so which variant's
    choices were scored can only be carried by the function's identity.

    gold_indices(doc) returns the indices of the correct choices; the optional
    postprocess(metrics, predictions, doc) can add task-specific entries.
    """
    prefix = "process_"
    if not attribute_name.startswith(prefix):
        raise AttributeError(attribute_name)
    yaml_path = os.path.join(directory, attribute_name[len(prefix) :] + ".yaml")
    if not os.path.isfile(yaml_path):
        raise AttributeError(f"{attribute_name}: no such config {yaml_path}")
    with open(yaml_path) as fh:
        doc_to_choice = yaml.load(fh, Loader=_ConfigLoader).get("doc_to_choice")
    if doc_to_choice is None:
        raise ValueError(f"{yaml_path} does not define doc_to_choice")

    def process_results(doc, results):
        choices = resolve_choices(doc_to_choice, doc)
        metrics, predictions = score_choices(results, choices, gold_indices(doc))
        if postprocess is not None:
            postprocess(metrics, predictions, doc)
        return metrics

    return process_results


def split_loglikelihoods(results, n_choices):
    """Split the flat result list into conditional and unconditional loglikelihoods.

    The harness sends one loglikelihood request per answer choice. When
    acc_mutual_info is in the metric list, it appends one extra unconditional
    ("", choice) request per choice after the conditional ones. It does not tell
    process_results which case occurred, so the doubled length is the only signal
    that the unconditional loglikelihoods (the PMI denominators) are present.
    """
    lls = [result[0] if isinstance(result, (tuple, list)) else result for result in results]
    if len(lls) not in (n_choices, 2 * n_choices):
        raise ValueError(
            f"Expected {n_choices} or {2 * n_choices} loglikelihoods, got {len(lls)}; "
            "do the choices passed to score_choices match the task's doc_to_choice?"
        )
    return lls[:n_choices], lls[n_choices:]


def softmax(scores):
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    return [exp / total for exp in exps]


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
        metrics[f"prob_correct{suffix}"] = sum(probs[i] for i in gold_indices)
    return metrics, predictions
