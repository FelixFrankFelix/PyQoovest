import math
import numpy as np
import pandas as pd
import requests
import app.config as config
#import config as config #test
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
#from app.utils.disease import disease_dic
#from app.utils.fertilizer import fertilizer_dic
from app.utils.model import ResNet9
import app.exceptions as exceptions
#from utils.model import ResNet9 #test
#import exceptions as exceptions #test
import app.recommedation_engine as rec
#import recommedation_engine as rec

fertilizer = pd.read_csv("app/Data/fertilizer.csv")
fertilizer_dict = dict(fertilizer)
planting = pd.read_csv("app/Data/crop-planting-info.csv")
planting_dict = dict(planting)


# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

disease_model_path = 'app/models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Loading crop recommendation model

crop_recommendation_model_path = 'app/models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# =========================================================================================

# Custom functions for calculations

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label with probabilities
    :params: img (image bytes)
    :params: model (pretrained model)
    :return: dictionary with predicted probabilities for each class
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(yb, dim=1)
    # Convert probabilities to a dictionary
    probabilities_dict = {disease_classes[i]: probabilities[0][i].item() for i in range(len(disease_classes))}

    return probabilities_dict

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def calc_scale(actual_number, input_number, k):
    absolute_difference = abs(actual_number - input_number)
    scale_value = 100 * math.exp(-k * absolute_difference**2)
    return scale_value

 # Convert potential numpy types to native Python types
def convert_to_python_type(value):
    if isinstance(value, (np.integer, np.int_)):
        return int(value)
    if isinstance(value, (np.float_, np.float32, np.float64)):
        return float(value)
    return value

def predict_crop_service(nitrogen, phosphorous, potassium, ph, rainfall, city):
    weather = weather_fetch(city)
    if weather:
        temperature, humidity = weather
        data = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        crop_name = my_prediction[0]
    
        fertilizer_data = fertilizer
        planting_data = planting

        crop_name = my_prediction[0]
        N_normal = fertilizer_data[fertilizer_data["Crop"] == crop_name]["N"]
        P_normal = fertilizer_data[fertilizer_data["Crop"] == crop_name]["P"]
        K_normal = fertilizer_data[fertilizer_data["Crop"] == crop_name]["K"]
        pH_normal = fertilizer_data[fertilizer_data["Crop"] == crop_name]["pH"]

        N_scale = calc_scale(nitrogen,N_normal,0.002)
        P_scale = calc_scale(phosphorous,P_normal,0.002)
        K_scale = calc_scale(potassium,K_normal,0.002)
        pH_scale = calc_scale(ph,pH_normal,0.2)

        best_season = planting_data[planting_data["crop"] == crop_name]["best_season"].values[0]
        season_start = planting_data[planting_data["crop"] == crop_name]["season_start"].values[0]
        season_end = planting_data[planting_data["crop"] == crop_name]["season_end"].values[0]
        days_to_maturity = planting_data[planting_data["crop"] == crop_name]["days_to_maturity"].values[0]
        
        planting_data = {
            "best_season": str(best_season),
            "season_start": int(season_start),
            "season_end": int(season_end),
            "days_to_maturity": int(days_to_maturity)
        }
        #print(planting_data)
        result = {
        "crop": crop_name,
        "nitrogen": nitrogen,
        "phosphorous": phosphorous,
        "potassium": potassium,
        "temp": temperature,
        "humidity": humidity,
        "ph": ph,
        "nitrogen_normal": float(N_normal),
        "phosphorous_normal": float(P_normal),
        "potassium_normal": float(K_normal),
        "ph_normal": float(pH_normal),
        "nitrogen_scale": N_scale,
        "phosphorous_scale": P_scale,
        "potassium_scale": K_scale,
        "ph_scale": pH_scale,
        "planting_data": planting_data
        }
        return {
        "responseCode": exceptions.ResponseConstant.SUCCESS.responseCode,
        "responseMessage": exceptions.ResponseConstant.SUCCESS.responseMessage,
        "body": result
        }

    else:
        return {
        "responseCode": exceptions.ResponseConstant.INVALID_ENTRY.responseCode,
        "responseMessage": exceptions.ResponseConstant.INVALID_ENTRY.responseMessage, 
        }

def predict_fertilizer_service(cropname,nitrogen,phosphorous,potassium):
    crop_name = cropname.lower()
    soil_N = nitrogen
    soil_P = phosphorous
    soil_K = potassium
    
    df = fertilizer

    crop_list = list(df['Crop'].unique())
    if crop_name not in crop_list:
        return {
        "responseCode": exceptions.ResponseConstant.INVALID_ENTRY.responseCode,
        "responseMessage": exceptions.ResponseConstant.INVALID_ENTRY.responseMessage, 
        }

    margin = 0.2
    # Extract the required levels for the specified crop
    N_normal = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_normal = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_normal = df[df['Crop'] == crop_name]['K'].iloc[0]

    # Define the range for Normal levels
    def get_level(soil_value, crop_value):
        lower_bound = crop_value * (1 - margin)
        upper_bound = crop_value * (1 + margin)
        if soil_value > upper_bound:
            return "High"
        elif soil_value < lower_bound:
            return "Low"
        else:
            return "Normal"

    # Determine the levels for N, P, and K
    soil_N_level = get_level(soil_N, N_normal)
    soil_P_level = get_level(soil_P, P_normal)
    soil_K_level = get_level(soil_K, K_normal)

    # Convert potential numpy types to native Python types
    def convert_to_python_type(value):
        if isinstance(value, (np.integer, np.int_)):
            return int(value)
        if isinstance(value, (np.float_, np.float32, np.float64)):
            return float(value)
        return value

    N_scale = calc_scale(nitrogen,N_normal,0.05)
    P_scale = calc_scale(phosphorous,P_normal,0.05)
    K_scale = calc_scale(potassium,K_normal,0.05)
    # Create the response dictionary
    data = {
        'crop': crop_name,
        'nitrogen': convert_to_python_type(soil_N),
        'phosphorous': convert_to_python_type(soil_P),
        'potassium': convert_to_python_type(soil_K),
        'nitrogen_normal': convert_to_python_type(N_normal),
        'phosphorous_normal': convert_to_python_type(P_normal),
        'potassium_normal': convert_to_python_type(K_normal),
        'nitrogen_level': soil_N_level,
        'phosphorous_level': soil_P_level,
        'potassium_level': soil_K_level,
        'nitrogen_scale': N_scale,
        'phosphorous_scale': P_scale,
        'potassium_scale': K_scale
    }
    return {
        "responseCode": exceptions.ResponseConstant.SUCCESS.responseCode,
        "responseMessage": exceptions.ResponseConstant.SUCCESS.responseMessage,
        "body": data
        }

def get_crops_fertilizer():
    crops = list(fertilizer["Crop"].unique())
    return {
        "responseCode": exceptions.ResponseConstant.SUCCESS.responseCode,
        "responseMessage": exceptions.ResponseConstant.SUCCESS.responseMessage,
        "body": crops
        }

def get_crops_diseases():
    # Extract unique crop names
    unique_crops = set()
    for disease_class in disease_classes:
        crop = disease_class.split("___")[0]
        unique_crops.add(crop)

    # Convert the set to a sorted list (optional)
    unique_crops_list = sorted(unique_crops)

    return {
        "responseCode": exceptions.ResponseConstant.SUCCESS.responseCode,
        "responseMessage": exceptions.ResponseConstant.SUCCESS.responseMessage,
        "body": unique_crops_list
        }

def get_crop_recommendation_service(factor,factor_value,factor_normal,crop_name):
    recommendation_result,exception_status = rec.factor_crop_rec(factor, factor_value,factor_normal,crop_name)
    #print(recommendation_result,exception_status)
    if exception_status == "NO":
        return {
        "responseCode": exceptions.ResponseConstant.SUCCESS.responseCode,
        "responseMessage": exceptions.ResponseConstant.SUCCESS.responseMessage,
        "body": recommendation_result
         }
    elif exception_status == "YES":
        return {
        "responseCode": exceptions.ResponseConstant.ERROR_PROCESSING.responseCode,
        "responseMessage": recommendation_result
         }

def get_fertilizer_recommendation_service(nitrogen,phosphorous,potassium,nitrogen_level,phosphorous_level,
                                            potassium_level,nitrogen_normal,phosphorous_normal,potassium_normal,crop_name):
    
    recommendation_result,exception_status = rec.factor_fert_rec(
        nitrogen,
        phosphorous,
        potassium,
        nitrogen_level,
        phosphorous_level,
        potassium_level,
        nitrogen_normal,
        phosphorous_normal,
        potassium_normal,
        crop_name
    )
    
    if exception_status == "NO":
        return {
        "responseCode": exceptions.ResponseConstant.SUCCESS.responseCode,
        "responseMessage": exceptions.ResponseConstant.SUCCESS.responseMessage,
        "body": recommendation_result
         }
    elif exception_status == "YES":
        return {
        "responseCode": exceptions.ResponseConstant.ERROR_PROCESSING.responseCode,
        "responseMessage": recommendation_result
         }

def get_disease_recommendation_service(crop_name,disease_name):
    
    recommendation_result,exception_status = rec.factor_disease_rec(
        crop_name, disease_name
    )
    
    if exception_status == "NO":
        return {
        "responseCode": exceptions.ResponseConstant.SUCCESS.responseCode,
        "responseMessage": exceptions.ResponseConstant.SUCCESS.responseMessage,
        "body": recommendation_result
         }
    elif exception_status == "YES":
        return {
        "responseCode": exceptions.ResponseConstant.ERROR_PROCESSING.responseCode,
        "responseMessage": recommendation_result
         }

def get_best_crop_prediction(response, crop):
    """
    Filters the response to only include results related to the specified crop
    and returns the crop disease with the highest predicted probability.
    
    :param response: Dictionary containing the predicted probabilities for each crop disease.
    :param crop: The crop to filter predictions by.
    :return: Dictionary with the best result for the specified crop.
    """
    crop = crop.lower()

    # Filter predictions related to the specified crop
    filtered_predictions = {k: v for k, v in response.items() if k.lower().startswith(crop + "___")}

    # Find the crop disease with the highest probability
    if not filtered_predictions:
        return None
    
    best_crop_disease = max(filtered_predictions, key=filtered_predictions.get)
    best_proba = filtered_predictions[best_crop_disease]

    return {"crop_condition": best_crop_disease, "pred_proba": best_proba}

def disease_prediction_service(img,crop_name):
    prediction = predict_image(img)
    result = get_best_crop_prediction(prediction, crop_name)

    if result["pred_proba"] < 0.3:
        return {
        "responseCode": exceptions.ResponseConstant.ERROR_PROCESSING.responseCode,
        "responseMessage": "Incorrect Crop Input or Incorrect Image Format"
        }
    elif result["pred_proba"] < 0.7:
        return {
        "responseCode": exceptions.ResponseConstant.LOW_CONFIDENCE.responseCode,
        "responseMessage": exceptions.ResponseConstant.LOW_CONFIDENCE.responseMessage,
        "body": result
        }
    else:
        return {
        "responseCode": exceptions.ResponseConstant.SUCCESS.responseCode,
        "responseMessage": exceptions.ResponseConstant.SUCCESS.responseMessage,
        "body": result
        }
   
#print(planting.info())
#print(predict_crop_service(10,10,10,7,100,"Lagos"))