class BLEUEvaluator:
    def __init__(self):
        try:
            import sacrebleu
            self._sacrebleu = sacrebleu
        except ImportError:
            raise ImportError("sacrebleu is required.")

    def compute(self, hypotheses, references) -> float:
        assert len(hypotheses) == len(references), \
            "hypotheses and references must have equal length."

        bleu = self._sacrebleu.corpus_bleu(
            hypotheses,
            [references],
        )
        return round(bleu.score, 4)