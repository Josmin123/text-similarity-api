from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model
from similarity import compute_similarity

app=FastAPI()
model=load_model()

class TextPair(BaseModel):
    text1:str
    text2:str

@app.post("/")   
def get_similarity(pair:TextPair):
    score=compute_similarity(pair.text1,pair.text2,model)
    return {"similarity score": score}


