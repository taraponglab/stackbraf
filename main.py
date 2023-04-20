from fastapi import FastAPI
from pydantic import BaseModel
import stackbraf

class Prediction(BaseModel):
    name: str
    smiles: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "StackBRAF Model"}

@app.post("/stackbraf/")
async def model(prediction: Prediction):
    result = stackbraf.execute_algorithm(prediction.smiles, prediction.name)
    return result
