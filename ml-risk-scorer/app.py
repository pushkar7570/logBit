from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

class PolicySnippet(BaseModel):
    text: str
    similarity: float | None = 0.0
    sectionRef: str | None = None

class ScoreRequest(BaseModel):
    alertText: str
    policySnippets: List[PolicySnippet] = []
    features: Dict[str, Any] = {}
    ruleHit: bool | None = None

class ScoreResponse(BaseModel):
    zeroShot: Dict[str, float]
    anomaly: float
    policySimMax: float
    policyRefs: List[str]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/score", response_model=ScoreResponse)
async def score(body: ScoreRequest):
    serious = 0.7 if "login failed" in body.alertText.lower() else 0.2
    policy_sim_max = max([s.similarity or 0.0 for s in body.policySnippets] + [0.0])
    refs = [s.sectionRef for s in body.policySnippets if s.sectionRef]
    return ScoreResponse(
        zeroShot={"serious": serious, "benign": 1 - serious},
        anomaly=0.0,
        policySimMax=policy_sim_max,
        policyRefs=refs,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)