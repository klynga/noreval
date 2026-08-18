"""ERRANT scoring for `ask_gec_nob`, run inside lm-evaluation-harness.

ERRANT is a corpus-level metric: it aligns every source/target and
source/prediction pair into M2 files and compares them in a single pass. That
maps onto the harness's passthrough-metric pattern used by `bleu`/`chrf` in
`lm_eval/api/metrics.py` -- `process_results` emits one tuple per document and
`errant_agg` receives the whole list and scores it once.

The harness computes no stderr for custom aggregations, so the standard error
is provided as the companion metric `errant_f05_stderr`: ERRANT still runs
exactly once (the corpus pass is cached on the item content), per-sentence
TP/FP/FN counts are recovered from the two M2 files, verified against
`errant_compare`'s corpus score, and bootstrapped over documents
(arXiv:2411.00640 -- questions are the sampling unit, so a document's K
sampled corrections are resampled together).

The subprocess calls and the prediction normalisation mirror `errant.py` in this
directory: predictions are normalised with `.replace("\\n\\n", "\\n")` and
nothing else. An empty prediction stays empty, is written as a blank line, and
is scored as deleting the sentence -- a model that generates nothing must not
be credited with recognising an already-correct sentence.

"""

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_sm"

# The console scripts are thin wrappers around these entry points, so calling
# the module directly is equivalent to calling the script.
ERRANT_COMMANDS = {
    "errant_parallel": "errant.commands.parallel_to_m2",
    "errant_compare": "errant.commands.compare_m2",
}


def process_results(doc, results):
    """Emit the per-document triple that `errant_agg` scores as a corpus.

    Normalisation matches `ask_gec_nob/errant.py`: collapse blank lines and nothing
    else. An empty prediction stays empty, so ERRANT scores it as deleting the
    sentence rather than as declining to correct it -- a model that generates
    nothing should not be credited with recognising a correct sentence.
    """
    triples = [
        (doc["source"], doc["correction"], prediction.replace("\n\n", "\n"))
        for prediction in results[0]
    ]
    return {"errant_f05": triples, "errant_f05_stderr": triples}


BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 1234

# the corpus pass is expensive (spaCy parses every sentence), and the harness
# aggregates `errant_f05` and `errant_f05_stderr` separately over equal item
# lists -- cache the corpus result on the item content so ERRANT runs once
_CORPUS_CACHE = {}


def errant_agg(items):
    """Corpus-level ERRANT F0.5 over every document in the task.

    Returns NaN rather than raising so that a broken ERRANT installation costs
    a single metric instead of the whole run's results, which the harness
    aggregates only after every task has finished generating.
    """
    try:
        return _errant_corpus(items)[0]
    except Exception as exc:
        logger.error("ERRANT evaluation failed for ask_gec_nob: %s", exc)
        return float("nan")


def errant_f05_stderr(items):
    """Question-level bootstrap standard error of the corpus F0.5.

    Documents (with their K sampled corrections) are resampled with
    replacement; each replicate's F0.5 is recomputed from the pooled
    per-sentence TP/FP/FN counts recovered from the M2 files.
    """
    try:
        _, doc_counts = _errant_corpus(items)
        if doc_counts is None:
            return float("nan")
        import random
        import statistics

        rng = random.Random(BOOTSTRAP_SEED)
        n = len(doc_counts)
        replicates = []
        for _ in range(BOOTSTRAP_ITERS):
            tp = fp = fn = 0
            for i in (rng.randrange(n) for _ in range(n)):
                tp += doc_counts[i][0]
                fp += doc_counts[i][1]
                fn += doc_counts[i][2]
            replicates.append(_f05(tp, fp, fn))
        return statistics.stdev(replicates)
    except Exception as exc:
        logger.error("ERRANT stderr bootstrap failed for ask_gec_nob: %s", exc)
        return float("nan")


def _f05(tp, fp, fn):
    # computeFScore in errant's compare_m2: an empty side counts as perfect
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if 0.25 * p + r else 0.0


def _edit_sets(m2_text):
    """Per-sentence sets of (start, end, correction) edits from an M2 file.

    `-lev` alignment writes a single annotator per sentence, and
    `errant_compare`'s default mode scores span-based correction, so an edit's
    identity is exactly this triple; `noop` entries mark edit-free sentences
    and are not edits.
    """
    sentences = []
    for block in m2_text.strip("\n").split("\n\n"):
        edits = set()
        for line in block.split("\n"):
            if not line.startswith("A "):
                continue
            span, etype, correction = line[2:].split("|||")[:3]
            if etype == "noop":
                continue
            start, end = span.split()
            edits.add((int(start), int(end), correction))
        sentences.append(edits)
    return sentences


def _errant_corpus(items):
    key = hash(tuple(triple for doc_triples in items for triple in doc_triples))
    if key not in _CORPUS_CACHE:
        _CORPUS_CACHE[key] = _run_errant(items)
    return _CORPUS_CACHE[key]


def _run_errant(items):
    # each document contributes its K sampled corrections; the corpus is pooled
    # over samples, so every source sentence appears K times
    doc_sizes = [len(doc_triples) for doc_triples in items]
    flat = [triple for doc_triples in items for triple in doc_triples]
    sources, targets, predictions = (list(field) for field in zip(*flat))
    _check_no_newlines(predictions)

    parallel = _command("errant_parallel")
    compare = _command("errant_compare")
    _check_spacy_model()

    tmp_dir = tempfile.mkdtemp(prefix="ask_gec_nob_errant_")
    try:
        paths = {
            name: _write_lines(tmp_dir, name, lines)
            for name, lines in (
                ("sources", sources),
                ("targets", targets),
                ("predictions", predictions),
            )
        }
        m2 = {}
        for name in ("targets", "predictions"):
            m2[name] = os.path.join(tmp_dir, f"{name}.m2")
            _run(
                parallel
                + [
                    "-orig", paths["sources"],
                    "-cor", paths[name],
                    "-out", m2[name],
                    "-lev",
                    "-tok",
                ],
                "errant_parallel",
            )
        output = _run(
            compare + ["-ref", m2["targets"], "-hyp", m2["predictions"]],
            "errant_compare",
        )
        with open(m2["targets"], encoding="utf-8") as f:
            ref_edits = _edit_sets(f.read())
        with open(m2["predictions"], encoding="utf-8") as f:
            hyp_edits = _edit_sets(f.read())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    corpus_f05 = _parse_f05(output)

    # per-sentence counts, grouped back into per-document sums for the bootstrap
    counts = [
        (len(hyp & ref), len(hyp - ref), len(ref - hyp))
        for hyp, ref in zip(hyp_edits, ref_edits, strict=True)
    ]
    doc_counts, offset = [], 0
    for size in doc_sizes:
        doc = counts[offset : offset + size]
        doc_counts.append(tuple(sum(c[i] for c in doc) for i in range(3)))
        offset += size

    # self-check: the recovered counts must reproduce errant_compare's number
    # (it prints 4 decimals); otherwise report the corpus score without a stderr
    tp, fp, fn = (sum(c[i] for c in doc_counts) for i in range(3))
    if abs(_f05(tp, fp, fn) - corpus_f05) > 1e-3:
        logger.error(
            "recovered per-sentence ERRANT counts give F0.5=%.4f but "
            "errant_compare reports %.4f; skipping the bootstrap stderr",
            _f05(tp, fp, fn),
            corpus_f05,
        )
        return corpus_f05, None

    return corpus_f05, doc_counts


def _check_no_newlines(predictions):
    """ERRANT's M2 format is line-aligned, so an embedded newline shifts every
    subsequent sentence against its source. `generation_kwargs.until` should
    prevent this; fail loudly rather than report a silently misaligned score."""
    offenders = [i for i, pred in enumerate(predictions) if "\n" in pred]
    if offenders:
        raise ValueError(
            f"{len(offenders)} prediction(s) contain a newline and would "
            f"misalign the M2 files (doc indices: {offenders[:10]})"
        )


def _command(name):
    """Return the argv prefix that runs an ERRANT command.

    Prefers the console script next to the running interpreter, so the call
    stays inside the same environment regardless of PATH. Under Apptainer the
    console scripts can end up outside the image's bin directory and off PATH
    even though the package is installed, so fall back to invoking the same
    entry point through the interpreter itself. That only requires `errant` to
    be importable, which is the thing that actually matters.
    """
    candidate = Path(sys.executable).resolve().parent / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return [str(candidate)]

    found = shutil.which(name)
    if found is not None:
        return [found]

    if importlib.util.find_spec("errant") is None:
        raise FileNotFoundError(
            f"`{name}` is not on PATH and `errant` is not importable from "
            f"{sys.executable}. ERRANT is required to evaluate the model on "
            "ask_gec_nob: https://github.com/chrisjbryant/errant"
        )
    logger.info(
        "`%s` not found as a script; invoking %s through %s instead.",
        name,
        ERRANT_COMMANDS[name],
        sys.executable,
    )
    return [sys.executable, "-c", f"from {ERRANT_COMMANDS[name]} import main; main()"]


def _check_spacy_model():
    import spacy

    if not spacy.util.is_package(SPACY_MODEL):
        raise OSError(
            f"the spaCy model `{SPACY_MODEL}` is required by ERRANT but is not "
            f"installed. Run `python -m spacy download {SPACY_MODEL}`."
        )


def _write_lines(tmp_dir, name, lines):
    """Terminate every line, rather than joining with newlines.

    ERRANT reads the parallel files with `zip` over file handles. Iterating a
    file yields nothing for a trailing empty line, so `"\\n".join(...)` loses
    the last sentence whenever it is empty -- the prediction M2 then has one
    fewer sentence than the reference M2 and `errant_compare` asserts.
    Terminating every line makes iteration yield exactly len(lines) entries for
    all three files, whatever their contents.
    """
    fpath = os.path.join(tmp_dir, f"{name}.txt")
    with open(fpath, "w+", encoding="utf-8") as out:
        out.writelines(f"{line}\n" for line in lines)
    return fpath


def _run(cmd, name):
    """Force UTF-8 in the child so ERRANT can read Norwegian text on nodes
    where the locale would otherwise select ASCII."""
    env = {**os.environ, "PYTHONUTF8": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"`{name}` exited {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def _parse_f05(output):
    return float(output.strip().split("\n")[-2].split()[-1].strip())
