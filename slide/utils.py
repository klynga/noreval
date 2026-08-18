# NOTE: doc_to_choice is defined per prompt-variant yaml (slide_p*, slide_cf_p*,
# slide_hybrid_p* use the word choices below; slide_mcf_p* uses the letter choices).
# process_results has no access to the task config, so the choice lists are
# duplicated here and must be kept in sync with the corresponding yaml files.
WORD_CHOICES = ["bokmål", "nynorsk", "dansk", "svensk", "et annet språk"]
LETTER_CHOICES = ["A", "B", "C", "D", "E"]


def _process_multilabel(doc, results, choices):
    num_choices = len(choices)
    lls, _ = zip(*results, strict=True)

    # metric_list includes acc_mutual_info, so construct_requests appends one
    # unconditional ("", choice) request per choice after the conditional ones.
    if len(lls) == 2 * num_choices:
        lls_cond, lls_uncond = lls[:num_choices], lls[num_choices:]
    elif len(lls) == num_choices:
        lls_cond, lls_uncond = lls, None
    else:
        raise ValueError(
            f"Expected {num_choices} or {2 * num_choices} loglikelihoods, got {len(lls)}"
        )

    correct_indices = doc["correct_indices"]
    completion_len = [len(c) for c in choices]

    pred = max(range(num_choices), key=lambda i: lls_cond[i])
    pred_norm = max(range(num_choices), key=lambda i: lls_cond[i] / completion_len[i])

    result = {
        "acc": int(pred in correct_indices),
        "acc_norm": int(pred_norm in correct_indices),
        # Char-length-normalized loglikelihood of the best gold choice, mirroring
        # ConfigurableTask's built-in multi-target norm_loglikelihood_corr (task.py).
        "norm_loglikelihood_corr": (
            max(lls_cond[g] / completion_len[g] for g in correct_indices)
            if correct_indices
            else float("nan")
        ),
    }

    if lls_uncond is not None:
        mutual_info = [c - u for c, u in zip(lls_cond, lls_uncond, strict=True)]
        pred_mi = max(range(num_choices), key=lambda i: mutual_info[i])
        result["acc_mutual_info"] = int(pred_mi in correct_indices)

    return result


def process_multilabel_word(doc, results):
    return _process_multilabel(doc, results, WORD_CHOICES)


def process_multilabel_letter(doc, results):
    return _process_multilabel(doc, results, LETTER_CHOICES)
