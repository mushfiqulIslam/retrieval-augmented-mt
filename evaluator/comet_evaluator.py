import logging

import sacrebleu
import torch

logger = logging.getLogger(__name__)


class COMETEvaluator:
    """
    COMET is a learned metric that correlates better with human judgement
    than BLEU, especially for morphologically rich languages like Finnish.
    """
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da"):
        self.model_name   = model_name
        self._comet_model = None
        self.metric_name  = "comet"
        self._init()

    def _init(self):
        try:
            from comet import download_model, load_from_checkpoint
            logger.info(f"Loading COMET model: {self.model_name}")
            path = download_model(self.model_name)
            self._comet_model = load_from_checkpoint(path)
            logger.info("COMET model loaded.")
        except Exception as e:
            logger.warning(
                f"COMET not available ({e}). "
                "Falling back to chrF. Install with: pip install unbabel-comet"
            )
            self.metric_name = "chrF"
            self._comet_model = None

    def compute(self, hypotheses, references, sources,) -> float:
        if self._comet_model is not None:
            data = [
                {"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)
            ]
            gpus = 1 if torch.cuda.is_available() else 0
            output = self._comet_model.predict(data, batch_size=8, gpus=gpus)
            scores = output.scores if hasattr(output, "scores") else output[0]
            return round(float(sum(scores) / len(scores)), 4)
        else:
            chrf = sacrebleu.corpus_chrf(hypotheses, [references])
            return round(chrf.score, 4)