import logging

from context_selector.scorer import build_scorer
from context_selector.sentence_segmenter import SentenceSegmenter

logger = logging.getLogger(__name__)


class FilteredContextSelector:
    """
    System C context selector: score sentences, select top-N.
    """

    def __init__(self, cfg):
        self.cfg        = cfg
        self.segmenter  = SentenceSegmenter(method=cfg.sentence_splitter)
        self.scorer     = build_scorer(cfg)
        logger.info(
            f"FilteredContextSelector initialized: "
            f"scorer={cfg.scoring_method}, splitter={cfg.sentence_splitter}"
        )

    def select(self, source, retrieved_docs, top_n):
        """
        Select the top-N most relevant sentences from retrieved documents.
        """
        all_sentences = []
        for doc in retrieved_docs:
            sents = self.segmenter.split(doc["text"])
            all_sentences.extend(sents)

        if not all_sentences:
            return [], []

        scores = self.scorer.score(source, all_sentences)
        scored = list(zip(all_sentences, scores))
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        selected = [sent for sent, _ in scored_sorted[:top_n]]

        return selected, scored_sorted

    def build_context_string(self, selected_sentences, separator= " ") -> str:
        return separator.join(selected_sentences)