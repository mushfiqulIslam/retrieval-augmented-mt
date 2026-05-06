from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional
import json
import os


@dataclass
class RetrieverConfig:
    """BM25 retriever settings — frozen after setup."""
    method: str = "bm25"          # "bm25" | "dense"
    methods: List[str] = field(default_factory=lambda: ["bm25"])
    top_k_values: List[int] = field(default_factory=lambda: [3, 5])

    # BM25 hyperparameters (Okapi BM25 defaults)
    bm25_k1: float = 1.5
    bm25_b: float  = 0.75

    # Dense retriever (used when method="dense")
    dense_model: str = "all-MiniLM-L6-v2"
    dense_batch_size: int = 32


@dataclass
class ContextSelectorConfig:
    """Context selection settings for System C."""
    # Sentence scoring method "embedding" | "cross_encoder" | "lexical"
    scoring_method: str = "embedding"
    embedding_model: str = "all-MiniLM-L6-v2"
    # Top-N values for ablation study
    top_n_values: List[int] = field(default_factory=lambda: [1, 3, 5])
    # Sentence segmentation
    sentence_splitter: str = "nltk"     # "nltk" | "spacy"
    # Maximum context tokens to inject (model hard-limit guard)
    max_context_tokens: int = 400


@dataclass
class TranslatorConfig:
    """Translation model settings: must be identical across all systems."""
    model_name: str = "Helsinki-NLP/opus-mt-en-fi"
    # Generation hyperparameters — fixed for all experiments
    num_beams: int = 4
    max_length: int = 256
    early_stopping: bool = True
    batch_size: int = 8
    # Context injection separator
    context_separator: str = " ||| "


@dataclass
class DataConfig:
    """Dataset settings."""
    # HuggingFace dataset for EN-FI parallel test set
    test_source: str = "hf_dataset"  # "hf_dataset" | "builtin"
    dataset_name: str = "Helsinki-NLP/opus-100"
    dataset_config: str = "en-fi"
    test_size: int = 200

    # Retrieval corpus: separate English documents
    corpus_source: str = "hf_dataset"
    corpus_file: Optional[str] = None
    corpus_size: int = 500


@dataclass
class EvaluationConfig:
    """Evaluation metric settings."""
    compute_bleu: bool = True
    compute_comet: bool = True
    comet_model: str = "Unbabel/wmt22-comet-da"
    compute_hallucination: bool = True
    spacy_model: str = "en_core_web_sm"
    compute_context_efficiency: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    seed: int = 42
    output_dir: str = "./results"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    save_translations: bool = True
    save_scores: bool = True
    run_system_a: bool = True   # MT-Only baseline
    run_system_b: bool = True   # RAG-Naïve
    run_system_c: bool = True   # RAG-Filtered
    run_ablations: bool = True  # N=1,3,5; random; full-doc

    retriever:         RetrieverConfig        = field(default_factory=RetrieverConfig)
    context_selector:  ContextSelectorConfig  = field(default_factory=ContextSelectorConfig)
    translator:        TranslatorConfig       = field(default_factory=TranslatorConfig)
    data:              DataConfig             = field(default_factory=DataConfig)
    evaluation:        EvaluationConfig       = field(default_factory=EvaluationConfig)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)

        nested_keys = {"retriever", "context_selector", "translator", "data", "evaluation"}
        top_level = {
            field.name: data[field.name]
            for field in fields(cls)
            if field.name in data and field.name not in nested_keys
        }
        cfg = cls(**top_level)
        cfg.retriever        = RetrieverConfig(**data.get("retriever", {}))
        if "retriever" in data and "methods" not in data["retriever"]:
            cfg.retriever.methods = [cfg.retriever.method]
        cfg.context_selector = ContextSelectorConfig(**data.get("context_selector", {}))
        cfg.translator       = TranslatorConfig(**data.get("translator", {}))
        cfg.data             = DataConfig(**data.get("data", {}))
        cfg.evaluation       = EvaluationConfig(**data.get("evaluation", {}))
        return cfg
