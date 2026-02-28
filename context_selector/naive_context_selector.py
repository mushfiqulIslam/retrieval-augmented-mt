class NaiveContextSelector:
    """
    System B context selector: concatenate full retrieved documents.
    No filtering, no truncation beyond model hard limits.
    """
    def __init__(self, max_tokens=400):
        self.max_tokens = max_tokens

    def select(self, retrieved_docs) -> str:
        texts = [doc["text"] for doc in retrieved_docs]
        context = " ".join(texts)
        words = context.split()
        if len(words) > self.max_tokens:
            context = " ".join(words[:self.max_tokens])

        return context