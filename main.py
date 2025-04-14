from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from service import process_input

app = FastAPI(title="Medical Code Predictor")

class PredictionRequest(BaseModel):
    Complaints: list[dict] = []
    investigations: list[dict] = []
    MedicalAdvice: list[dict] = []

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        print(request.dict())
        response = process_input(request.dict())
        return {"status": "success", "result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
