from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import torch

from typing import List

from training.models import CustomResNet50
from utils_app import pipeline_prediction
import os #<--- this

app = FastAPI()

# load model
# checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoints/CustomResNet50/ckp_epoch_2.pth") #<--- this
checkpoint = torch.load('./checkpoints/CustomResNet50/ckp_epoch_2.pth')
NN_MODEL= CustomResNet50(pretrained=False)
NN_MODEL.load_state_dict(checkpoint['model_state_dict'])

class ImageRequest(BaseModel):
    img_raw: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: int

pipeline = pipeline_prediction(NN_MODEL)

@app.get("/")
async def root():
    return {"message": "Starting FastAPI digit detection app"}


@app.post("/predict")
def predict(image: ImageRequest) -> PredictionResponse:
    data = image.img_raw    
    data = pipeline.handle_missing_val(np.array(data))
    data = pipeline.preprocess_image(np.array(data))
    pred = pipeline.make_predictions(data)
    return PredictionResponse(prediction=pred)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)