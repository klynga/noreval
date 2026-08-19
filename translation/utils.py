import os
import sys

# noreval_corpus.py lives at the repository root, one level up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from noreval_corpus import bleu_stderr, chrf_stderr, pooled_bleu, pooled_chrf  # noqa: F401
