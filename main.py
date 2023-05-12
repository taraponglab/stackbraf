from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from tasks import stackbraf_model

class StackBRAFPredictor(BaseModel):
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
async def StackBRAFModel(prediction: StackBRAFPredictor):
    process = stackbraf_model.delay(prediction.smiles, prediction.name)
    return {"process_id": process.id}

@app.get("/stackbraf/{process_id}")
async def StackBRAFResult(process_id: str):
    result = stackbraf_model.AsyncResult(process_id)
    if (result.successful()):
        return {"status": "complete",
                "result": result.get(timeout=1)}
    else:
        return {"status": "processing"}
