from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import app.transformer_model as tm
import app.sentence_similarity as sm

app = FastAPI()


class Item(BaseModel):
    text: str


@app.post("/sentiment/predict_many/")
def model_predict(item: Item):
    res = tm.predict(item.text)
    return {"sentiment": res}


@app.post("/sentiment/predict_one/")
def model_predict_one(item: Item):
    res = tm.predict_one(item.text)
    return {"sentiment": res}


@app.post("/sentiment/weighted_embedding/")
def sw_emb(item: Item):
    res = sm.sentiment_weighted_text_embedding(item.text)
    return {"sentiment": list(res)}


@app.post("/sentiment/average_embedding/")
def av_emb(item: Item):
    res = sm.averaged_text_embedding(item.text)
    return {"sentiment": list(res)}
