# 🇳🇴 NorEval

The up-to-date NorEval config files for [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The main branch is the official source for **NorEval version 1.1**, supporting all 35 tasks used in our [Norwegian LLM dashboard](https://ltgoslo.github.io/llm-dashboard).

_____

**Paper title:** [NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark](https://aclanthology.org/2025.findings-acl.181/)

NorEval is a multi-task Norwegian language understanding and generation evaluation benchmark that combines existing peer-reviewed datasets with new datasets manually created from scratch. NorEval covers diverse task categories: sentiment analysis, Norwegian language knowledge, Norwegian-specific \& world knowledge, machine reading comprehension, commonsense reasoning, machine translation, text summarization, instruction following, and truthfulness. Our main evaluation principles are:

- 🌐 **Linguistic diversity**: support for both of the official written standards of Norwegian: Bokmål and Nynorsk (the minority variant).
- 📊 **Task diversity**: coverage of various least addressed tasks for Norwegian. In particular, only three out of 24 NorEval datasets are included in existing Norwegian benchmarks to date: [NorBench](https://aclanthology.org/2023.nodalida-1.61/), [NLEBench](https://aclanthology.org/2024.emnlp-main.317/), [ScandEval](https://aclanthology.org/2023.nodalida-1.20/), and [SEB](https://proceedings.neurips.cc/paper_files/paper/2024/file/4746bb91bd073ec7eef930d5775122ba-Paper-Datasets_and_Benchmarks_Track.pdf).
- 🧠 **Data quality**: focus on only peer-reviewed human-created datasets to ensure reliable evaluation in the context of the Norwegian language, culture, and values.
- 📏 **Prompt sensitivity**: evaluation across multiple human-written prompts to account for the prompt sensitivity.
- 👩🏻‍🔬 **Standardized evaluation**: integration of NorEval into LM Evaluation Harness for flexible and reproducible evaluation.

_____

### Tasks

|Name  |Bokmål | Nynorsk  | Northern Sámi | Task type  | Task category |
|:---|:---|:---|:---|:---|:---|
|[NoReC Sentence](https://huggingface.co/datasets/ltg/norec_sentence) |`norec_sentence`  | — | — |Text classification| Sentiment analysis |
|[NoReC Document](https://huggingface.co/datasets/ltg/norec_document) |`norec_document`  | — | — |Text classification| Sentiment analysis |
|[SLIDE](https://huggingface.co/datasets/ltg/slide) |`slide`  | — | — |Text classification| Norwegian language knowledge |
|[NCB](https://huggingface.co/datasets/hcfa/ncb) |`ncb`| — | — |Sentence ranking| Norwegian language knowledge   |
|[NoCoLA](https://huggingface.co/datasets/ltg/nocola) |`nocola`| — | — |Sentence ranking| Norwegian language knowledge   |
|[MultiBLiMP](https://huggingface.co/datasets/jumelet/multiblimp) | — | — |`noreval_multiblimp`|Sentence ranking| Northern Sámi language knowledge |
|[NorIdiom](https://huggingface.co/datasets/Sprakbanken/Norwegian_idioms) |`noridiom_nob`  | `noridiom_nno`  | — |Sentence completion| Norwegian language knowledge  |
|[NorBelebele](https://huggingface.co/datasets/ltg/norbelebele) |`norbelebele`| —| — |Multiple-choice question answering| Machine reading comprehension |
|[NRK-Quiz-QA](https://huggingface.co/datasets/ltg/nrk_quiz_qa) |`nrk_quiz_qa_nob`| `nrk_quiz_qa_nno`| — |Multiple-choice question answering| Norwegian-specific & world knowledge |
|[NorOpenBookQA](https://huggingface.co/datasets/ltg/noropenbookqa) |`noropenbookqa_nob`| `noropenbookqa_nno` | — |Multiple-choice question answering| Norwegian-specific & world knowledge |
|[NorOpenBookQA (without fact)](https://huggingface.co/datasets/ltg/noropenbookqa) |`noropenbookqa_no_fact_nob`| `noropenbookqa_no_fact_nno` | — |Multiple-choice question answering| Norwegian-specific & world knowledge |
|[NorCommonsenseQA](https://huggingface.co/datasets/ltg/norcommonsenseqa) |`norcommonsenseqa_nob`| `norcommonsenseqa_nno` | — |Multiple-choice question answering|Commonsense reasoning  |
|[NorTruthfulQA Multiple choice](https://huggingface.co/datasets/ltg/nortruthfulqa_mc) |`nortruthfulqa_mc_nob`| `nortruthfulqa_mc_nno` | — |Multiple-choice question answering |Truthfulness |
|[NorQuAD](https://huggingface.co/datasets/ltg/norquad) |`norquad`| —  | — |Generative question answering |Machine reading comprehension |
|[NorTruthfulQA Generation](https://huggingface.co/datasets/ltg/nortruthfulqa_gen) |`nortruthfulqa_gen_nob`| `nortruthfulqa_gen_nno` | — | Generative question answering|Truthfulness |
|[ASK-GEC](https://huggingface.co/datasets/ltg/ask-gec) |`ask_gec`| — | — |Sequence-to-sequence generation|Norwegian language knowledge |
|[NorSumm](https://huggingface.co/datasets/SamiaT/NorSumm)  |`norsumm_nob` | `norsumm_nno`  | — |Sequence-to-sequence generation|Text summarization |
|[NorSumm Translation (Bokmål → Nynorsk)](https://huggingface.co/datasets/ltg/norsumm-nob-nno-translation) | `norsumm_nob_nno_translation`| —  | — |Sequence-to-sequence generation|Machine translation |
|[NorSumm Translation (Nynorsk → Bokmål)](https://huggingface.co/datasets/ltg/norsumm-nob-nno-translation) | — | `norsumm_nno_nob_translation`  | — |Sequence-to-sequence generation|Machine translation |
|[Tatoeba (English → Bokmål/Nynorsk)](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | `tatoeba_eng_nob`| `tatoeba_eng_nno`  | — |Sequence-to-sequence generation|Machine translation |
|[Tatoeba (Bokmål/Nynorsk → English)](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | `tatoeba_nob_eng`| `tatoeba_nno_eng`  | — |Sequence-to-sequence generation|Machine translation |
|[Tatoeba (Bokmål → Northern Sámi)](https://huggingface.co/datasets/ltg/saami-tatoeba) | — | — | `tatoeba_nob_sme`|Sequence-to-sequence generation|Machine translation |
|[Tatoeba (Northern Sámi → Bokmål)](https://huggingface.co/datasets/ltg/saami-tatoeba) | — | — | `tatoeba_sme_nob`|Sequence-to-sequence generation|Machine translation |
|[NorRewrite-Instruct](https://huggingface.co/datasets/ltg/norrewrite-instruct) |`norrewrite_instruct`  |— | — |Sequence-to-sequence generation|Instruction following|
|[NorSummarize-Instruct](https://huggingface.co/datasets/ltg/norsummarize-instruct) |`norsummarize_instruct` |— | — |Sequence-to-sequence generation|Instruction following|

<details open>
<summary><b>Table description</b></summary>

* **Name**: a dataset name with a HuggingFace link.
* **Bokmål**: the LM Evaluation Harness task name for the Norwegian Bokmål dataset. `slide`, which covers Bokmål, Nynorsk, Danish, and Swedish, is also listed here.
* **Nynorsk**: the LM Evaluation Harness task name for the Norwegian Nynorsk dataset, if available.
* **Northern Sámi**: the LM Evaluation Harness task name for the Northern Sámi dataset, if available. Translation tasks between Bokmål and Northern Sámi are listed here.
* **Task type**: the task type.
* **Task category**: the task category.

</details>

_____

### How to use

The tasks are plain [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) configs — no fork of the harness is needed. Install the harness, clone this repository, and point the harness to it with `--include_path`:

```bash
pip install lm-eval  # or lm-eval[vllm] for the vLLM backend
git clone https://github.com/klynga/noreval.git
```

Example: zero-shot evaluation on NorQuAD across all five prompts:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=norallm/normistral-7b-warm \
  --include_path noreval \
  --tasks norquad \
  --output results/norquad/0-shot/ \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

* The task names in the table above run all prompt variants of a dataset at once; append a prompt index to run a single variant (e.g. `norquad_p2`).
* Tasks loaded via `--include_path` take precedence over any same-named NorEval tasks bundled with the harness, so this repository is the source of truth for the task definitions.
* `ask_gec` does not compute a metric inside the harness: run it with `--log_samples` and score the generated corrections externally with [ERRANT](https://github.com/chrisjbryant/errant) (F0.5).

_____

### Citation

```bibtex
@article{mikhailov2025noreval,
  title={NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark},
  author={Mikhailov, Vladislav and Enstad, Tita and Samuel, David and Farseth{\aa}s, Hans Christian and Kutuzov, Andrey and Velldal, Erik and {\O}vrelid, Lilja},
  journal={arXiv preprint arXiv:2504.07749},
  year={2025}
}
```
