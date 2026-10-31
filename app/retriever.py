from dataclasses import dataclass

from app.embeddings import EmbeddingService
from app.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    distance: float


class Retriever:
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_service.embed_query(query)

        result = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            chunks.append(
                RetrievedChunk(
                    text=document,
                    source=metadata.get("source", "unknown"),
                    page=metadata.get("page", 0),
                    distance=float(distance),
                )
            )

        return chunks