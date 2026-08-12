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
    return [doc["choices"]["label"].index(doc["answer"])]


def __getattr__(name):
    # utils.process_<variant> scores the doc_to_choice of <variant>.yaml,
    # so new prompt variants need no code changes here
    return variant_process_results(_here, name, _gold_indices)


def p0_nb(doc):
    prompt = "Spørsmål: {question}\n\nSvar:"
    return prompt.format(question=doc["question"])


def p1_nb(doc):
    prompt = "{question}\n\nSvaralternativer:{choices}\n\nHva er riktig svar?\n\nSvar:"
    choices = "".join(list(map(lambda choice: f"\n- {choice}", doc["choices"]["text"])))
    return prompt.format(question=doc["question"], choices=choices)


def p2_nb(doc):
    prompt = (
        "{question}{choices}\n\nEr det riktige svaret {enumerated_choices}?\n\nSvar:"
    )
    choices = "".join(
        [
            f"\n{label}: {option}"
            for label, option in zip(doc["choices"]["label"], doc["choices"]["text"])
        ]
    )
    enumerated_choices = ", ".join(
        doc["choices"]["label"][:-1]
    ) + ", eller {latest_choice}".format(latest_choice=doc["choices"]["label"][-1])
    if len(doc["choices"]["label"]) == 2:
        enumerated_choices = enumerated_choices.replace(", eller", " eller")
    return prompt.format(
        question=doc["question"], choices=choices, enumerated_choices=enumerated_choices
    )


def p3_nb(doc):
    prompt = "Spørsmål: {question}{choices}\n\nSvar:"
    choices = "".join(
        [
            f"\n{label}: {option}"
            for label, option in zip(doc["choices"]["label"], doc["choices"]["text"])
        ]
    )
    return prompt.format(question=doc["question"], choices=choices)


def p4_nb(doc):
    prompt = "{question}\nVelg riktig svar blant disse alternativene:{choices}\n\nSvar:"
    choices = "".join(list(map(lambda choice: f"\n- {choice}", doc["choices"]["text"])))
    return prompt.format(question=doc["question"], choices=choices)
