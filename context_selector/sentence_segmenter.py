import logging
import re

import nltk
import spacy
from nltk import sent_tokenize

logger = logging.getLogger(__name__)


class SentenceSegmenter:
    def __init__(self, method= "nltk"):
        self.method = method
        self._init_segmenter()

    def _init_segmenter(self):
        if self.method == "nltk":
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            try:
                self._tokenize = sent_tokenize
                logger.debug("Sentence segmenter: NLTK punkt.")
            except ImportError:
                logger.warning("NLTK not available. Falling back to regex sentence splitter.")
                self._tokenize = self._regex_split

        elif self.method == "spacy":
            try:
                self._nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
                self._nlp.add_pipe("sentencizer")
                self._tokenize = self._spacy_split
                logger.debug("Sentence segmenter: spaCy sentencizer.")
            except Exception:
                logger.warning("spaCy not available. Falling back to regex.")
                self._tokenize = self._regex_split
        else:
            self._tokenize = self._regex_split

    @staticmethod
    def _regex_split(text):
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _spacy_split(self, text):
        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def split(self, text):
        sentences = self._tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]