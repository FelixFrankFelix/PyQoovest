# app/app_fastapi.py

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import requests
import app.config as config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
#from app.utils.disease import disease_dic
#from app.utils.fertilizer import fertilizer_dic
from app.utils.model import ResNet9
from fastapi.middleware.cors import CORSMiddleware
import app.service as service
import app.schema as schema
import app.recommedation_engine as rec
#from mangum import Magnum



app = FastAPI()
#handler = Magnum(app)

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ===============================================================================================
# ------------------------------------ FASTAPI ENDPOINTS -----------------------------------------

@app.get("/")
async def home():
    return {"STATUS":"OK"}


@app.post("/crop-predict",response_model=schema.ResponseUnionCropPrediction)
async def crop_prediction(request: schema.CropPredictionRequest):
    result = service.predict_crop_service(
        request.nitrogen,
        request.phosphorous,
        request.potassium,
        request.ph,
        request.rainfall,
        request.city
    )
    return result

@app.post("/fertilizer-predict",response_model=schema.ResponseUnionFertilizerPrediction)
async def fert_recommend(request: schema.FertilizerPredictionRequest):
    data = service.predict_fertilizer_service(
        request.cropname,
        request.nitrogen,
        request.phosphorous,
        request.potassium
    )
    return data

@app.get("/get-crops-fertilizer",response_model=schema.GetCropsResponse)
async def get_crops_fertilizer():
    result = service.get_crops_fertilizer()
    return result

@app.get("/get-crops-diseases",response_model=schema.GetCropsResponse)
async def get_crops_diseases():
    result = service.get_crops_diseases()
    return result

@app.post("/disease-predict/{crop_name}",response_model=schema.ResponseUnionDiseaseResponse)
async def disease_prediction(request: Request,crop_name:str, file: UploadFile = File(...)):
    #title = 'Harvestify - Disease Detection'
    img = await file.read()
    prediction = service.disease_prediction_service(img,crop_name)
    return prediction

@app.post("/get-crop-recommendation",response_model=schema.ResponseUnionRecommendation)
async def get_crop_recommendation(request: schema.CropRecommedationRequest):
    recommendation_result = service.get_crop_recommendation_service(request.factor, request.factor_value,request.factor_normal,request.crop_name)
    return recommendation_result

@app.post("/get-ferterlizer-recommendation",response_model=schema.ResponseUnionRecommendation)
async def get_fertilizer_recommendation(request: schema.FertilzerRecommedationRequest):
    ferterlizer_recommendation = service.get_fertilizer_recommendation_service(
        request.nitrogen,
        request.phosphorous,
        request.potassium,
        request.nitrogen_level,
        request.phosphorous_level,
        request.potassium_level,
        request.nitrogen_normal,
        request.phosphorous_normal,
        request.potassium_normal,
        request.crop_name
    )
    return ferterlizer_recommendation

@app.post("/get-disease-recommendation",response_model=schema.ResponseUnionRecommendation)
async def get_disease_recommendation(request: schema.DiseaseRecommedationRequest):
    disease_recommendation = service.get_disease_recommendation_service(
        request.crop_name,
        request.disease_name
    )
    return disease_recommendation