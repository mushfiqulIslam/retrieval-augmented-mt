import logging
import time

import torch

from utils.config import TranslatorConfig

logger = logging.getLogger(__name__)


class Translator:
    """
    Wrapper around Helsinki-NLP/opus-mt-en-fi (MarianMT).
    """
    def __init__(self, cfg: TranslatorConfig):
        self.cfg = cfg
        self._model     = None
        self._tokenizer = None
        self._device    = None
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        logger.info(f"Loading translation model: {self.cfg.model_name}")
        start = time.perf_counter()

        self._tokenizer = MarianTokenizer.from_pretrained(self.cfg.model_name)
        self._model     = MarianMTModel.from_pretrained(self.cfg.model_name)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model  = self._model.to(self._device)
        self._model.eval()  # inference mode

        elapsed = time.perf_counter() - start
        logger.info(
            f"Model loaded on {self._device} in {elapsed:.2f}s. "
            f"Parameters: num_beams={self.cfg.num_beams}, max_length={self.cfg.max_length}."
        )

    def _build_input(self, source, context):
        """
        Build the model input string.

        If context is provided, prepend it to the source with a separator.
        Format: "{context} ||| {source}"

        The separator ||| is a common convention in context-aware NMT and is
        unlikely to appear in natural text, making it a clean boundary marker.
        """
        if context and context.strip():
            return f"{context.strip()}{self.cfg.context_separator}{source.strip()}"
        return source.strip()

    def translate(self, source, context=None):
        return self.translate_batch([source], [context] if context else [None])[0]

    def translate_batch(self, sources, contexts=None):
        if contexts is None:
            contexts = [None] * len(sources)

        assert len(sources) == len(contexts), \
            "sources and contexts must have the same length."

        inputs_text = [
            self._build_input(src, ctx)
            for src, ctx in zip(sources, contexts)
        ]

        all_translations = []

        # Process in mini-batches
        for i in range(0, len(inputs_text), self.cfg.batch_size):
            batch = inputs_text[i:i + self.cfg.batch_size]

            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._tokenizer.model_max_length,
            ).to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **encoded,
                    num_beams=self.cfg.num_beams,
                    max_length=self.cfg.max_length,
                    early_stopping=self.cfg.early_stopping,
                )

            decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            all_translations.extend(decoded)

        return all_translations

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, truncation=False))

    @property
    def model_name(self) -> str:
        return self.cfg.model_name