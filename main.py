import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model import My_Rec_Model
import uvicorn

PRETRAINED_CHECKPOINT_DIR = "./pretrained_checkpoint"
PRETRAINED_CHECKPOINT = os.path.join(PRETRAINED_CHECKPOINT_DIR, "best_mae_epoch_9.ckpt")

app = FastAPI()
with open("dataset/movie2idx.json") as f:
    movie2idx = json.load(f)
with open("dataset/idx2movie.json") as f:
    idx2movie = json.load(f)

model = My_Rec_Model(
    checkpoint_path=PRETRAINED_CHECKPOINT,
    movie2idx=movie2idx,
    idx2movie=idx2movie
)


class TrainRequest(BaseModel):
    dataset_path: str


class EvaluateRequest(BaseModel):
    dataset_path: str


class PredictRequest(BaseModel):
    movie_names: List[str]
    ratings: List[float]
    top_k: int = 20


class SimilarRequest(BaseModel):
    movie_name: str
    top_k: Optional[int] = 10


@app.post("/train")
def train():
    try:
        model.train()
        return {"status": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
def evaluate():
    try:
        model.evaluate()
        return {"status": "Evaluation completed (check logs)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = model.predict(req.movie_names, req.ratings, req.top_k)
        return {"recommended_ids": result[0], "estimated_ratings": result[1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar")
def similar(req: SimilarRequest):
    try:
        result = model.similar(req.movie_name, req.top_k)
        return {"similar_movies": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Recommender system API is up"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
