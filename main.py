import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from qa_engine import get_answer

app = FastAPI(title="Fast response Q&A API", description="QA system using FAISS, Groq, and Redis")

class QuestionRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    content: str
    source: str

class AnswerResponse(BaseModel):
    answer: str
    source_chunks: list[SourceChunk]
    response_time: float
    cached: bool

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = get_answer(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run uvicorn without workers to avoid Windows multiprocessing issues with HuggingFace models
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
