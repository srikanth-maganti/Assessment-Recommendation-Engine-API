from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from assessment_api import find_matches
from gen_model import run
from most_accurate import  find_most_accurate  # Your final LLM filter logic

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to SHL Assessment Recommendation Engine!"}
    

# Request model
class QueryRequest(BaseModel):
    job_query: str
# Response model
class Assessment(BaseModel):
    Assessment_Name: str
    Link: str
    Duration: str
    Remote_Testing: str
    Adaptive_Testing: str
    Language: str


@app.post("/debug")
def debug_endpoint(data: QueryRequest):
    matches = find_matches(data.job_query)
    return {"matches": matches[:3]} 


@app.post("/summarize", response_model=List[Assessment])
def search_assessments(data: QueryRequest):
    query = data.job_query
    matches = find_matches(query)  # embedding-based shortlist
    best_matches = find_most_accurate(matches, query)  # LLM-ranked top N

    return best_matches
