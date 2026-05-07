import platform
import sys
from datetime import datetime, timezone
from typing import Dict

import torch


VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}


def resolve_device(requested: str = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"Unknown device {requested!r}. Choose one of: {', '.join(sorted(VALID_DEVICES))}."
        )

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Device 'cuda' was requested, but CUDA is not available.")

    if requested == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            raise ValueError("Device 'mps' was requested, but Apple MPS is not available.")

    return requested


def build_run_metadata(cfg, resolved_device: str) -> Dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "requested_device": cfg.device,
        "resolved_device": resolved_device,
        "seed": cfg.seed,
        "models": {
            "translator": cfg.translator.model_name,
            "retriever_methods": cfg.retriever.methods,
            "dense_retriever": cfg.retriever.dense_model,
            "embedding_scorer": cfg.context_selector.embedding_model,
            "comet": cfg.evaluation.comet_model if cfg.evaluation.compute_comet else None,
            "spacy": cfg.evaluation.spacy_model if cfg.evaluation.compute_hallucination else None,
        },
    }
