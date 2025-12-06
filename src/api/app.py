from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.engine import SearchEngine

app = FastAPI()
engine = SearchEngine()
engine.load_index()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

@app.post("/search")
def search(query: Query):
    results = engine.search(query.text, k=5)
    return {"results": results}

