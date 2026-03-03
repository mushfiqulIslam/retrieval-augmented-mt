# retrieval-augmented-mt (RAG-MT)

Research-oriented experiment runner for **English → Finnish** machine translation with retrieval-augmented context and context selection.

The code evaluates three systems:

- **System A (MT-Only)**: baseline MarianMT translation with no retrieved context.
- **System B (RAG-Naïve)**: retrieves top-k English documents and injects the full retrieved text as context.
- **System C (RAG-Filtered)**: retrieves top-k documents, segments them into sentences, scores sentences against the source sentence, and injects only the best N sentences (context selection).

It also runs ablations (e.g., random sentence selection) and reports BLEU, COMET (or chrF fallback), a lightweight hallucination heuristic, and context efficiency (quality per injected token).

---

## Repository layout

- `run_experiments.py` — CLI entrypoint.
- `systems/` — orchestration + implementations of Systems A/B/C.
- `retriever/` — BM25 and dense retrieval + caching wrapper.
- `context_selector/` — sentence segmentation and context scoring/selection.
- `translator/` — MarianMT wrapper (Helsinki-NLP/opus-mt-en-fi).
- `evaluator/` — BLEU, COMET/chrF, hallucination heuristic, efficiency.
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

### Optional (recommended)

- **COMET** metric: `unbabel-comet` (if not installed, the code falls back to **chrF** automatically).

### Model/data downloads

On first run, the following will be downloaded automatically:

- Hugging Face dataset: `Helsinki-NLP/opus-100` (config `en-fi`)
- Translation model: `Helsinki-NLP/opus-mt-en-fi`
- Sentence embeddings model(s) (default: `all-MiniLM-L6-v2`)

NLTK `punkt` is downloaded automatically by the sentence segmenter.

If you enable spaCy sentence splitting or hallucination evaluation (default), install the English spaCy model:

```bash
python -m spacy download en_core_web_sm