from pydantic import BaseModel
from typing import Optional,Union,List
from datetime import datetime

class RecRequest(BaseModel):
    factor : str
    factor_value : float
    factor_normal : float
    crop_name: str

class FertRecRequest(BaseModel):
    N: float
    P: float
    K: float
    N_level: str
    P_level: str
    K_level: str
    N_normal: float
    P_normal: float
    K_normal: float
    crop_name: str