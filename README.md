# retrieval-augmented-mt (RAG-MT)

Research-oriented experiment runner for **English → Finnish** machine translation with retrieval-augmented context and context selection.

The code evaluates three systems:

- **System A (MT-Only)**: baseline MarianMT translation with no retrieved context.
- **System B (RAG-Naïve)**: retrieves top-k English documents and injects the full retrieved text as context.
- **System C (RAG-Filtered)**: retrieves top-k documents, segments them into sentences, scores sentences against the source sentence, and injects only the best N sentences (context selection).

It also runs ablations (e.g., random sentence selection) and reports BLEU, COMET (or chrF fallback), a lightweight entity novelty heuristic, and context efficiency (quality per injected token).

---

## Repository layout

- `run_experiments.py` — CLI entrypoint.
- `systems/` — orchestration + implementations of Systems A/B/C.
- `retriever/` — BM25 and dense retrieval + caching wrapper.
- `context_selector/` — sentence segmentation and context scoring/selection.
- `translator/` — MarianMT wrapper (Helsinki-NLP/opus-mt-en-fi).
- `evaluator/` — BLEU, COMET/chrF, entity novelty heuristic, efficiency.
- `utils/` — config schema, data loading, corpus sampling, I/O helpers.

---

## Requirements

### Python

- Python **3.9+** recommended.

### Core Python packages
Install core Python packages:

```bash
pip install -r requirements.txt
```

On macOS with a `uv` virtual environment, use the macOS requirements file:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements-macos.txt
```

The default `requirements.txt` includes Linux/NVIDIA CUDA packages from the
environment used to generate it. Those packages do not provide macOS wheels.

If `pip3 install -r requirements.txt` reports an `externally-managed-environment`
error on Homebrew Python, it is using the system/Homebrew interpreter rather than
the project virtual environment. In that case, prefer `uv pip install ...` or
`python -m pip ...` from the activated `.venv`.

### Quick smoke run

Use built-in sample data, a tiny test set, and no COMET download:

```bash
python run_experiments.py --quick --device auto
```

To reproduce the presentation-style setup with the 20-document built-in corpus:

```bash
python run_experiments.py --profile presentation --retrievers bm25 dense
```

The resolved configuration is saved to `results/experiment_config.json`, and
runtime metadata such as Python, platform, torch version, selected device, model
names, seed, and timestamp is saved to `results/run_metadata.json`.

The runner also saves PDF-aligned analysis artifacts:

- `context_traces.jsonl` — retrieved documents, selected sentences, and sentence scores.
- `hypothesis_verdict.json` — H1/H2/H3 verdicts computed from actual scores.
- `experiment_report.md` — concise results narrative with insights and limitations.
- `length_bucket_analysis.json` — short/medium/long source-sentence analysis.

### Optional (recommended)

- **COMET** metric: `unbabel-comet` (if not installed, the code falls back to **chrF** automatically).
- **Entity novelty heuristic**: the previous hallucination-style output is a lightweight named-entity novelty signal, not a robust hallucination benchmark.

### Model/data downloads

On first run, the following will be downloaded automatically:

- Hugging Face dataset: `Helsinki-NLP/opus-100` (config `en-fi`)
- Translation model: `Helsinki-NLP/opus-mt-en-fi`
- Sentence embeddings model(s) (default: `all-MiniLM-L6-v2`)

NLTK `punkt` is downloaded automatically by the sentence segmenter.

If you enable spaCy sentence splitting or entity novelty evaluation (default), install the English spaCy model:

```bash
python -m spacy download en_core_web_sm
