import json
import os
from dataclasses import asdict


def save_translations(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "system":         r.system_name,
                "source":         r.source,
                "reference":      r.reference,
                "hypothesis":     r.hypothesis,
                "context":        r.context,
                "context_tokens": r.context_tokens,
            }, ensure_ascii=False) + "\n")

def save_scores(all_scores, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in all_scores], f, indent=2, ensure_ascii=False)

def print_results_table(all_scores, comet_name="COMET"):
    sep = "─" * 95
    print("\n" + "═" * 95)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("═" * 95)
    header = (
        f"  {'System':<35} {'BLEU':>8} {comet_name:>10} "
        f"{'Hall.Rate':>10} {'Ctx Tokens':>11} {'Qual/Tok':>10}"
    )
    print(header)
    print(sep)

    for s in all_scores:
        print(
            f"  {s.system_name:<35} "
            f"{s.bleu:>8.2f} "
            f"{s.comet:>10.4f} "
            f"{s.hallucination_rate:>10.4f} "
            f"{s.avg_context_tokens:>11.1f} "
            f"{s.quality_per_token:>10.4f}"
        )
    print("═" * 95)

def print_hallucination_examples(results, evaluator, system_name: str, n_examples: int = 3):
    print(f"Hallucination examples — {system_name}:")
    shown = 0
    for r in results:
        rate, detail = evaluator._hall_eval.compute_sentence_hallucination(
            r.source, r.hypothesis, r.context
        )
        if detail["hallucinated"] and shown < n_examples:
            print(f"    SRC : {r.source}")
            print(f"    HYP : {r.hypothesis}")
            print(f"    Hall: {detail['hallucinated']}")
            print()
            shown += 1
        if shown >= n_examples:
            break

def print_research_conclusions(all_scores):
    print("\n" + "═" * 70)
    print("  RESEARCH HYPOTHESIS VALIDATION")
    print("═" * 70)

    score_a = next((s for s in all_scores if s.system_name == "System_A"), None)
    if not score_a:
        return

    print(f"\n  Baseline (System A — MT Only): BLEU={score_a.bleu:.2f}")
    print()

    # Group by top_k
    for top_k in [3, 5]:
        score_b = next((s for s in all_scores if s.system_name == f"System_B_k{top_k}"), None)
        if not score_b:
            continue

        print(f"  ── k={top_k} ──────────────────────────────────────────")
        print(f"  System B (Naïve)     BLEU={score_b.bleu:.2f}  "
              f"Hall={score_b.hallucination_rate:.3f}  Tok={score_b.avg_context_tokens:.0f}")

        for top_n in [1, 3, 5]:
            score_c = next(
                (s for s in all_scores if s.system_name == f"System_C_k{top_k}_N{top_n}"), None
            )
            if not score_c:
                continue
            bleu_delta = score_c.bleu - score_b.bleu
            hall_delta = score_c.hallucination_rate - score_b.hallucination_rate
            tok_delta  = score_c.avg_context_tokens - score_b.avg_context_tokens
            print(
                f"  System C (N={top_n})        BLEU={score_c.bleu:.2f} ({bleu_delta:+.2f})  "
                f"Hall={score_c.hallucination_rate:.3f} ({hall_delta:+.3f})  "
                f"Tok={score_c.avg_context_tokens:.0f} ({tok_delta:+.0f})"
            )

        score_rand = next(
            (s for s in all_scores if s.system_name == f"Ablation_Random_k{top_k}_N3"), None
        )
        if score_rand:
            print(
                f"  Ablation (Random N=3) BLEU={score_rand.bleu:.2f}  "
                f"Hall={score_rand.hallucination_rate:.3f}  Tok={score_rand.avg_context_tokens:.0f}"
            )
        print()

    print("─────────────────────────────────────────────────────────")
    print("Hypothesis checks:")

    # Check hypothesis 1: C ≥ B ≥ A
    best_c = max(
        (s for s in all_scores if s.system_name.startswith("System_C")),
        key=lambda s: s.bleu, default=None
    )
    best_b = max(
        (s for s in all_scores if s.system_name.startswith("System_B")),
        key=lambda s: s.bleu, default=None
    )
    if best_c and best_b and score_a:
        h1 = best_c.bleu >= best_b.bleu >= score_a.bleu
        print(f"[{'✓' if h1 else '?'}] BLEU: C ({best_c.bleu:.2f}) ≥ B ({best_b.bleu:.2f}) ≥ A ({score_a.bleu:.2f})")

    if best_c and best_b:
        h2 = best_c.hallucination_rate <= best_b.hallucination_rate
        print(f"[{'✓' if h2 else '?'}] Hallucination: C ({best_c.hallucination_rate:.3f}) ≤ B ({best_b.hallucination_rate:.3f})")

    if best_c and best_b:
        h3 = best_c.avg_context_tokens <= best_b.avg_context_tokens
        print(f"[{'✓' if h3 else '?'}] Context tokens: C ({best_c.avg_context_tokens:.0f}) ≤ B ({best_b.avg_context_tokens:.0f})")

    print("═" * 70)