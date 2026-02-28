import logging

from evaluator.bleu_evaluator import BLEUEvaluator
from evaluator.comet_evaluator import COMETEvaluator
from evaluator.context_efficiency_evaluator import ContextEfficiencyEvaluator
from evaluator.hallucination_evaluator import HallucinationEvaluator
from utils.data import MetricScores

logger = logging.getLogger(__name__)


class MasterEvaluator:
    def __init__(
        self,
        compute_bleu= True,
        compute_comet= True,
        compute_hallucination= True,
        compute_efficiency= True,
        comet_model= "Unbabel/wmt22-comet-da",
        spacy_model= "en_core_web_sm",
    ):
        self.compute_bleu_flag = compute_bleu
        self.compute_comet_flag = compute_comet
        self.compute_hallucination_flag = compute_hallucination
        self.compute_efficiency_flag = compute_efficiency

        self._bleu_eval  = BLEUEvaluator() if compute_bleu else None
        self._comet_eval = COMETEvaluator(model_name=comet_model) if compute_comet else None
        self._hall_eval  = HallucinationEvaluator(spacy_model=spacy_model) \
                            if compute_hallucination else None
        self._eff_eval   = ContextEfficiencyEvaluator() if compute_efficiency else None

    def evaluate(self, results, system_name, metadata=None):
        hypotheses = [r.hypothesis for r in results]
        references = [r.reference  for r in results]
        sources    = [r.source     for r in results]
        contexts   = [r.context    for r in results]
        ctx_tokens = [r.context_tokens for r in results]

        scores = MetricScores(
            system_name=system_name,
            n_sentences=len(results),
            metadata=metadata or {},
        )

        # BLEU
        if self._bleu_eval:
            scores.bleu = self._bleu_eval.compute(hypotheses, references)
            logger.info(f"  [{system_name}] BLEU = {scores.bleu:.2f}")

        # COMET / chrF
        if self._comet_eval:
            scores.comet = self._comet_eval.compute(hypotheses, references, sources)
            scores.comet_metric_name = self._comet_eval.metric_name
            logger.info(f"  [{system_name}] {scores.comet_metric_name} = {scores.comet:.4f}")

        # Hallucination
        if self._hall_eval:
            avg_rate, _ = self._hall_eval.compute_corpus_hallucination(
                sources, hypotheses, contexts
            )
            scores.hallucination_rate = avg_rate
            logger.info(f"  [{system_name}] Hallucination rate = {scores.hallucination_rate:.4f}")

        # Context efficiency
        if self._eff_eval and any(t > 0 for t in ctx_tokens):
            eff = self._eff_eval.compute(ctx_tokens, scores.bleu)
            scores.avg_context_tokens = eff["avg_context_tokens"]
            scores.quality_per_token  = eff["quality_per_token"]
            logger.info(
                f"  [{system_name}] Avg context tokens = {scores.avg_context_tokens:.1f}, "
                f"quality/token = {scores.quality_per_token:.4f}"
            )

        return scores