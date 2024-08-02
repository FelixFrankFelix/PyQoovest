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


app = FastAPI()

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


@app.post("/crop-predict")
async def crop_prediction(request: Request, nitrogen: int = Form(...), phosphorous: int = Form(...), pottasium: int = Form(...), ph: float = Form(...), rainfall: float = Form(...), city: str = Form(...)):
    result = service.predict_crop_service(nitrogen,phosphorous,pottasium,ph,rainfall,city)
    return result

@app.post("/fertilizer-predict")
async def fert_recommend(request: Request, cropname: str = Form(...), nitrogen: int = Form(...), phosphorous: int = Form(...), pottasium: int = Form(...)):
    data = service.predict_fertilizer_service(cropname,nitrogen,phosphorous,pottasium)
    return data

@app.get("/get-crops")
async def get_crops():
    crops = list(service.fertilizer["Crop"].unique())
    return crops

@app.post("/disease-predict")
async def disease_prediction(request: Request, file: UploadFile = File(...)):
    #title = 'Harvestify - Disease Detection'
    img = await file.read()
    prediction = service.predict_image(img)
    return prediction

@app.post("/get-crop-recommendation")
async def get_crop_recommendation(request: schema.RecRequest):
    recommendation_result = rec.factor_crop_rec(request.factor, request.factor_value,request.factor_normal,request.crop_name)
    return recommendation_result

@app.post("/get-ferterlizer-recommendation")
async def get_ferterlizer_recommendation(request: schema.FertRecRequest):
    ferterlizer_recommendation = rec.factor_fert_rec(request.N,request.P,request.K,request.N_level,request.P_level,request.K_level,request.N_normal,request.P_normal,request.K_normal,request.crop_name)
    return ferterlizer_recommendation