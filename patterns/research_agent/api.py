"""
Production entrypoint: exposes the research agent as a REST API.

Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import research_companies
from schema import CompanyList

app = FastAPI(title="GCC Startup Research Agent")


class ResearchRequest(BaseModel):
    query: str


@app.post("/research", response_model=CompanyList)
def research(req: ResearchRequest) -> CompanyList:
    try:
        return research_companies(req.query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
