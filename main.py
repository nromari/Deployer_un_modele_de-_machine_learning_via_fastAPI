import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from predict_tags import predict_fct

class Application(BaseModel):
    question: str

class Decision(BaseModel):
    predicted_tags: list[str]

app = FastAPI()

@app.get("/")
def welcome():
    return "Post stackoverflow question"

@app.post("/application", response_model=Decision)
async def create_application(application: Application):
    question = application.question
    decision = {
        "predicted_tags" : predict_fct(question)
    }
    return decision

if __name__ == "__main__":
    uvicorn.run("main:app")
