from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import torch
from typing import List

from training.models import CustomResNet50
from utils_app import pipeline_prediction, sample_ddpm

app = FastAPI()

# load model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKP = torch.load('./checkpoints/CustomResNet50/ckp_epoch_20.pth', map_location=torch.device(DEVICE))
NN_MODEL= CustomResNet50(pretrained=False).to(DEVICE)
NN_MODEL.load_state_dict(CKP['model_state_dict'])

class ImageRequest(BaseModel):
    img_raw: List[List[float]]

class PredictionResponse(BaseModel):
    prediction: int

class NumberRequest(BaseModel):
    number: int
 
class ImageResponse(BaseModel):
    img_list: List[List[int]]

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

@app.post("/generate")
def generate(number: NumberRequest) -> ImageResponse:
    num = number.number
    img = sample_ddpm(num)
    img_list = img.cpu().numpy().tolist()
    return ImageResponse(img_list=img_list)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)