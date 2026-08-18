def process_results(doc, results):
    # results[0] holds the K sampled translations (repeats + take_first_k
    # filter); every sample is paired with the reference and pooled into the
    # corpus BLEU/chrF computed in ../utils.py
    pairs = [(doc["text_sme"], prediction) for prediction in results[0]]
    return {"bleu": pairs, "chrf": pairs, "bleu_stderr": pairs, "chrf_stderr": pairs}
