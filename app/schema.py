from pydantic import BaseModel
from typing import Optional,Union,List
from datetime import datetime

class BaseResponse(BaseModel):
    responseCode: str
    responseMessage: str

class CropRecommedationRequest(BaseModel):
    factor : str
    factor_value : float
    factor_normal : float
    crop_name: str

class FertilzerRecommedationRequest(BaseModel):
    nitrogen: float
    phosphorous: float
    potassium: float
    nitrogen_normal: float
    phosphorous_normal: float
    potassium_normal: float
    nitrogen_level: str
    phosphorous_level: str
    potassium_level: str
    crop_name: str

class DiseaseRecommedationRequest(BaseModel):
    crop_name: str
    disease_name: str


class PlantingData(BaseModel):
    best_season: str
    season_start: int
    season_end: int
    days_to_maturity: int

class CropPredictionRequest(BaseModel):
    nitrogen: int
    phosphorous: int
    potassium: int
    ph: float
    rainfall: float
    city: str


class CropPredictionResponse(BaseModel):
    crop: str
    nitrogen: int
    phosphorous: int
    potassium: int
    temp: float
    humidity: int
    ph: float
    nitrogen_normal: int
    phosphorous_normal: int
    potassium_normal: int
    ph_normal: float
    nitrogen_scale: float
    phosphorous_scale: float
    potassium_scale: float
    ph_scale: float
    planting_data: PlantingData

class FertilizerPredictionRequest(BaseModel):
    cropname: str
    nitrogen: int
    phosphorous: int
    potassium: int

class FertilizerPredictionResponse(BaseModel):
    crop: str
    nitrogen: float
    phosphorous: float
    potassium: float
    nitrogen_normal: float
    phosphorous_normal: float
    potassium_normal: float
    nitrogen_level: str
    phosphorous_level: str
    potassium_level: str
    nitrogen_scale: float
    phosphorous_scale: float
    potassium_scale: float

class DiseaseResponse(BaseModel):
    crop_condition: str
    pred_proba: float

class GetCropsResponse(BaseResponse):
    body: list

class GetRecommedationResponse(BaseResponse):
    body: str

class DetailedDiseaseResponse(BaseResponse):
    body: DiseaseResponse

class DetailedCropPredictionResponse(BaseResponse):
    body: CropPredictionResponse

class DetailedFertilizerPredictionResponse(BaseResponse):
    body: FertilizerPredictionResponse

ResponseUnionCropPrediction = Union[DetailedCropPredictionResponse, BaseResponse]
ResponseUnionFertilizerPrediction = Union[DetailedFertilizerPredictionResponse, BaseResponse]
ResponseUnionRecommendation = Union[GetRecommedationResponse, BaseResponse]
ResponseUnionDiseaseResponse = Union[DetailedDiseaseResponse, BaseResponse]