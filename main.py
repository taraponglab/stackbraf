from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import stackbraf

class Prediction(BaseModel):
    name: str
    smiles: str

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "StackBRAF Model"}

@app.post("/stackbraf/")
async def model(prediction: Prediction):
    result = stackbraf.execute_algorithm(prediction.smiles, prediction.name)
    return result
