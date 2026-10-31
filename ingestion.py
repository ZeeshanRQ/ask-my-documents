from pathlib import Path

from app.chunker import split_text
from app.document_loader import load_pdf
from app.embeddings import EmbeddingService
from app.vector_store import VectorStore


def ingest_pdf(file_path: str) -> None:
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    pages = load_pdf(file_path)

    all_ids: list[str] = []
    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    for page in pages:
        chunks = split_text(
            text=page["text"],
            chunk_size=300,
            overlap=50,
        )

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = (
                f"{page['source']}"
                f"-page-{page['page']}"
                f"-chunk-{chunk_index}"
            )

            all_ids.append(chunk_id)
            all_chunks.append(chunk)
            all_metadata.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": chunk_index,
                }
            )

    if not all_chunks:
        raise ValueError("No readable text was found in the PDF")

    embeddings = embedding_service.embed_documents(all_chunks)

    vector_store.add_documents(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )

    print(f"Successfully stored {len(all_chunks)} chunks.")


if __name__ == "__main__":
    pdf_path = Path("documents/sample_company_policy.pdf")
    ingest_pdf(str(pdf_path))