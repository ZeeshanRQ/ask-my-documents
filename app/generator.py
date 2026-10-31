from ollama import chat

from app.retriever import RetrievedChunk


class AnswerGenerator:
    def __init__(self, model: str = "qwen3:4b") -> None:
        self.model = model

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        context_parts: list[str] = []

        for index, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Source {index}: {chunk.source}, page {chunk.page}]\n"
                f"{chunk.text}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present, say:
   "I could not find this information in the provided documents."
3. Cite the source number used in the answer.
4. Keep the answer clear and factual.

Context:
{context}

Question:
{question}
""".strip()

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.message.content