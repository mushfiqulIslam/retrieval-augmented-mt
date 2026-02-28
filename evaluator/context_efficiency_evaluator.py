class ContextEfficiencyEvaluator:
    @staticmethod
    def compute(context_token_counts, bleu_score):
        avg_tokens = sum(context_token_counts) / len(context_token_counts) \
            if context_token_counts else 0.0
        total_tokens = sum(context_token_counts)
        quality_per_token = bleu_score / avg_tokens if avg_tokens > 0 else 0.0

        return {
            "avg_context_tokens":  round(avg_tokens, 2),
            "total_context_tokens": total_tokens,
            "quality_per_token":   round(quality_per_token, 4),
            "bleu_score":          bleu_score,
        }
