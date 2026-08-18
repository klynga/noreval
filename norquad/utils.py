import datasets
import transformers.data.metrics.squad_metrics as squad_metrics


def _score_one(doc, prediction):
    reference = doc["answers"]["text"][0]
    f1_sum = squad_metrics.compute_f1(reference, prediction)
    exact_match = squad_metrics.compute_exact(reference, prediction)
    return {"f1": f1_sum, "exact_match": exact_match}


def process_results(doc, results):
    # results[0] holds the K sampled answers (repeats + take_first_k filter);
    # the question-level score is the mean over samples (arXiv:2411.00640, §3.1)
    scored = [_score_one(doc, prediction) for prediction in results[0]]
    return {key: sum(s[key] for s in scored) / len(scored) for key in scored[0]}


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["title"] = doc["context"].strip().split("\n")[0].strip()
        doc["passage"] = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        doc["question"] = " ".join(doc["question"].strip().split())
        return doc

    return dataset.map(_helper)


def p0(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f"Tittel: {title}\n\nTekst: {passage}\n\nSpørsmål: {question}\n\nSvar:"
    return prompt


def p1(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nGitt teksten over, hva er svaret på følgende spørsmål? "{question}"\n\nSvar:'
    return prompt


def p2(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = (
        f"Tittel: {title}\n\nTekst: {passage}\n\nSvar på følgende: {question}\n\nSvar:"
    )
    return prompt


def p3(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nHvordan kan man svare på spørsmålet "{question}", gitt teksten over?\n\nSvar:'
    return prompt


def p4(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst:{passage}\n\nGitt teksten over, besvar følgende spørsmål: "{question}"\n\nSvar:'
    return prompt
