import json
import os
from typing import Dict, List, Optional

from evaluator.bleu_evaluator import BLEUEvaluator
from utils.data import MetricScores, TranslationResult


BUCKETS = {
    "short": (0, 10),
    "medium": (11, 20),
    "long": (21, None),
}


def _score_family(system_name: str) -> str:
    if system_name == "System_A":
        return "A"
    if system_name.startswith("System_B"):
        return "B"
    if system_name.startswith("System_C"):
        return "C"
    return "other"


def _best_score(scores: List[MetricScores], family: str, key: str) -> Optional[MetricScores]:
    candidates = [s for s in scores if _score_family(s.system_name) == family]
    if not candidates:
        return None
    return max(candidates, key=lambda s: getattr(s, key))


def build_hypothesis_verdict(scores: List[MetricScores]) -> Dict:
    baseline = next((s for s in scores if s.system_name == "System_A"), None)
    best_b = _best_score(scores, "B", "bleu")
    best_c = _best_score(scores, "C", "bleu")

    quality_scores = [s for s in [baseline, best_b, best_c] if s is not None]
    ranking = [s.system_name for s in sorted(quality_scores, key=lambda s: s.bleu, reverse=True)]

    h2_passed = None
    if best_b and best_c:
        h2_passed = best_c.hallucination_rate <= best_b.hallucination_rate

    h3_passed = None
    best_b_eff = _best_score(scores, "B", "quality_per_token")
    best_c_eff = _best_score(scores, "C", "quality_per_token")
    if best_b_eff and best_c_eff:
        h3_passed = best_c_eff.quality_per_token >= best_b_eff.quality_per_token

    return {
        "h1_quality_ranking": {
            "description": "Actual BLEU ranking among MT-only, best naive RAG, and best filtered RAG.",
            "ranking": ranking,
            "best_system": ranking[0] if ranking else None,
        },
        "h2_entity_novelty": {
            "description": "Filtered RAG should have lower or equal entity novelty than naive RAG.",
            "passed": h2_passed,
            "best_filtered_system": best_c.system_name if best_c else None,
            "best_naive_system": best_b.system_name if best_b else None,
            "filtered_rate": best_c.hallucination_rate if best_c else None,
            "naive_rate": best_b.hallucination_rate if best_b else None,
            "caveat": "Entity novelty is a lightweight heuristic, not a robust hallucination benchmark.",
        },
        "h3_context_efficiency": {
            "description": "Filtered RAG should achieve higher BLEU per context token than naive RAG.",
            "passed": h3_passed,
            "best_filtered_system": best_c_eff.system_name if best_c_eff else None,
            "best_naive_system": best_b_eff.system_name if best_b_eff else None,
            "filtered_quality_per_token": best_c_eff.quality_per_token if best_c_eff else None,
            "naive_quality_per_token": best_b_eff.quality_per_token if best_b_eff else None,
        },
    }


def _bucket_name(source: str) -> str:
    n_tokens = len(source.split())
    for name, (min_tokens, max_tokens) in BUCKETS.items():
        if n_tokens >= min_tokens and (max_tokens is None or n_tokens <= max_tokens):
            return name
    return "long"


def build_length_bucket_analysis(
    all_results: Dict[str, List[TranslationResult]],
    entity_rates: Optional[Dict[str, float]] = None,
    entity_evaluator=None,
) -> Dict:
    entity_rates = entity_rates or {}
    bleu_eval = BLEUEvaluator()
    analysis = {}

    for system_name, results in all_results.items():
        buckets = {name: [] for name in BUCKETS}
        for result in results:
            buckets[_bucket_name(result.source)].append(result)

        analysis[system_name] = {}
        for bucket_name, bucket_results in buckets.items():
            if bucket_results:
                refs = [r.reference for r in bucket_results]
                hyps = [r.hypothesis for r in bucket_results]
                try:
                    bleu = bleu_eval.compute(hyps, refs)
                except Exception:
                    bleu = 0.0
                avg_context_tokens = sum(r.context_tokens for r in bucket_results) / len(bucket_results)
                if entity_evaluator is not None:
                    rates = [
                        entity_evaluator.compute_sentence_hallucination(
                            r.source, r.hypothesis, r.context
                        )[0]
                        for r in bucket_results
                    ]
                    avg_entity_rate = sum(rates) / len(rates)
                else:
                    avg_entity_rate = entity_rates.get(system_name)
            else:
                bleu = 0.0
                avg_context_tokens = 0.0
                avg_entity_rate = entity_rates.get(system_name)

            analysis[system_name][bucket_name] = {
                "n_sentences": len(bucket_results),
                "bleu": round(bleu, 4),
                "avg_entity_novelty_rate": round(avg_entity_rate, 4) if avg_entity_rate is not None else None,
                "avg_context_tokens": round(avg_context_tokens, 2),
            }

    return analysis


def build_experiment_report(scores: List[MetricScores], verdict: Dict, length_analysis: Dict) -> str:
    best_by_bleu = sorted(scores, key=lambda s: s.bleu, reverse=True)
    best_by_efficiency = sorted(scores, key=lambda s: s.quality_per_token, reverse=True)

    lines = [
        "# RAG-MT Experiment Report",
        "",
        "## Insights",
        f"- Best BLEU system: {best_by_bleu[0].system_name if best_by_bleu else 'n/a'}.",
        f"- Best context efficiency system: {best_by_efficiency[0].system_name if best_by_efficiency else 'n/a'}.",
        "- Entity novelty is reported as a lightweight heuristic, not a robust hallucination benchmark.",
        "",
        "## Hypothesis Verdict",
        f"- H1 quality ranking: {', '.join(verdict['h1_quality_ranking']['ranking']) or 'n/a'}.",
        f"- H2 entity novelty passed: {verdict['h2_entity_novelty']['passed']}.",
        f"- H3 context efficiency passed: {verdict['h3_context_efficiency']['passed']}.",
        "",
        "## Best Configurations",
    ]

    for score in best_by_bleu[:5]:
        lines.append(
            f"- {score.system_name}: BLEU={score.bleu:.2f}, "
            f"EntNew={score.hallucination_rate:.4f}, "
            f"AvgCtxTok={score.avg_context_tokens:.1f}, "
            f"BLEU/Tok={score.quality_per_token:.4f}"
        )

    lines.extend([
        "",
        "## Length Bucket Analysis",
        f"- Systems analyzed: {len(length_analysis)}.",
        "- Buckets: short <= 10 tokens, medium 11-20 tokens, long >= 21 tokens.",
        "",
        "## Limitations",
        "- Built-in presentation corpus contains 20 documents, limiting relevant-context coverage.",
        "- Long context can distract the translation model from the source sentence.",
        "- Finnish morphology can change named-entity surface forms, affecting entity novelty counts.",
    ])

    return "\n".join(lines) + "\n"


def save_pdf_aligned_reports(
    output_dir: str,
    scores: List[MetricScores],
    all_results: Dict[str, List[TranslationResult]],
    entity_evaluator=None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    verdict = build_hypothesis_verdict(scores)
    entity_rates = {s.system_name: s.hallucination_rate for s in scores}
    length_analysis = build_length_bucket_analysis(
        all_results,
        entity_rates=entity_rates,
        entity_evaluator=entity_evaluator,
    )

    with open(os.path.join(output_dir, "hypothesis_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "length_bucket_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(length_analysis, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "experiment_report.md"), "w", encoding="utf-8") as f:
        f.write(build_experiment_report(scores, verdict, length_analysis))
