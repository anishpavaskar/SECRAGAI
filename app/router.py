from fastapi import APIRouter
from pydantic import BaseModel
from app.agent import query_rag

router = APIRouter()

class QueryRequest(BaseModel):
    q: str

@router.post("/query")
def rag_query(request: QueryRequest):
    return query_rag(request.q)
