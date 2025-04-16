from fastapi import FastAPI, HTTPException, Request
from service import process_input

app = FastAPI(title="Medical Code Predictor")

@app.post("/predict")
async def predict(request: Request):
    try:
        input_data = await request.json()
        print(input_data)
        response = process_input(input_data)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
