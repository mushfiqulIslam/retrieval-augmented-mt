import random
from typing import List

from context_selector.sentence_segmenter import SentenceSegmenter


class RandomContextSelector:
    """
    Ablation control: select N sentences at random (no relevance scoring).
    Used to isolate the contribution of relevance-based filtering.
    """

    def __init__(self, cfg, seed= 42):
        self.cfg       = cfg
        self.seed      = seed
        self.segmenter = SentenceSegmenter(method=cfg.sentence_splitter)

    def select(self, retrieved_docs, top_n) -> List[str]:
        all_sentences: List[str] = []
        for doc in retrieved_docs:
            sents = self.segmenter.split(doc["text"])
            all_sentences.extend(sents)

        if not all_sentences:
            return []

        rng = random.Random(self.seed)
        rng.shuffle(all_sentences)
        return all_sentences[:top_n]