from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.generator import AnswerGenerator
from app.retriever import Retriever


app = FastAPI(
    title="Ask My Documents",
    version="1.0.0",
)

retriever = Retriever()
generator = AnswerGenerator()


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
    )


class SourceResponse(BaseModel):
    source: str
    page: int
    distance: float
    text: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest) -> QuestionResponse:
    try:
        chunks = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant document chunks were found",
            )

        answer = generator.generate(
            question=request.question,
            chunks=chunks,
        )

        return QuestionResponse(
            answer=answer,
            sources=[
                SourceResponse(
                    source=chunk.source,
                    page=chunk.page,
                    distance=chunk.distance,
                    text=chunk.text,
                )
                for chunk in chunks
            ],
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to process question: {error}",
        ) from error