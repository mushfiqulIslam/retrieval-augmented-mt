import logging
import re

logger = logging.getLogger(__name__)


class HallucinationEvaluator:
    """
    NER-based hallucination detector.
    Metric (from spec):
      hallucination_rate = (entities in output NOT in source or context) / (total output entities)

    Named entities are extracted from the English source and context,
    and from the Finnish translation output.

    """
    def __init__(self, spacy_model= "en_core_web_sm"):
        self.spacy_model = spacy_model
        self._nlp = None
        self._init()

    def _init(self):
        try:
            import spacy
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError:
                logger.warning(
                    f"spaCy model '{self.spacy_model}' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                self._nlp = None
        except ImportError:
            logger.warning("spaCy not installed. Hallucination evaluation will use regex fallback.")
            self._nlp = None

    def _extract_entities_spacy(self, text):
        doc = self._nlp(text)
        return {ent.text.lower().strip() for ent in doc.ents
                if ent.label_ in {
                    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME",
                    "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
                }}

    @staticmethod
    def _extract_entities_regex(text):
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
        matches = re.findall(pattern, text)
        return {m.lower().strip() for m in matches if len(m) > 2}

    def extract_entities(self, text):
        if not text or not text.strip():
            return set()

        if self._nlp is not None:
            return self._extract_entities_spacy(text)

        return self._extract_entities_regex(text)

    def compute_sentence_hallucination(self, source, hypothesis, context=None):
        src_entities = self.extract_entities(source)
        ctx_entities = self.extract_entities(context) if context else set()
        known_entities = src_entities | ctx_entities

        hyp_entities = self.extract_entities(hypothesis)

        if not hyp_entities:
            return 0.0, {
                "source_entities":     sorted(src_entities),
                "context_entities":    sorted(ctx_entities),
                "output_entities":     [],
                "hallucinated":        [],
                "rate":                0.0,
            }

        hallucinated = set()
        for ent in hyp_entities:
            found = any(
                ent in known_ent or known_ent in ent
                for known_ent in known_entities
            )
            if not found:
                hallucinated.add(ent)

        rate = len(hallucinated) / len(hyp_entities)

        return rate, {
            "source_entities":    sorted(src_entities),
            "context_entities":   sorted(ctx_entities),
            "output_entities":    sorted(hyp_entities),
            "hallucinated":       sorted(hallucinated),
            "rate":               rate,
        }

    def compute_corpus_hallucination(self, sources, hypotheses, contexts=None):
        if contexts is None:
            contexts = [None] * len(sources)

        details = []
        rates   = []
        for src, hyp, ctx in zip(sources, hypotheses, contexts):
            rate, detail = self.compute_sentence_hallucination(src, hyp, ctx)
            rates.append(rate)
            details.append(detail)

        avg_rate = sum(rates) / len(rates) if rates else 0.0
        return round(avg_rate, 4), details