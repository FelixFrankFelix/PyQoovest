import math
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


fertilizer = pd.read_csv("app/Data/fertilizer.csv")
fertilizer_dict = dict(fertilizer)

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
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
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
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

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

def predict_crop_service(nitrogen, phosphorous, pottasium, ph, rainfall, city):
    weather = weather_fetch(city)
    if weather:
        temperature, humidity = weather
        data = np.array([[nitrogen, phosphorous, pottasium, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        crop_name = my_prediction[0]
    
        fertilizer_data = fertilizer

        crop_name = my_prediction[0]
        N_normal = fertilizer_data[fertilizer_data["Crop"] == crop_name]["N"]
        P_normal = fertilizer_data[fertilizer_data["Crop"] == crop_name]["P"]
        K_normal = fertilizer_data[fertilizer_data["Crop"] == crop_name]["K"]
        pH_normal = fertilizer_data[fertilizer_data["Crop"] == crop_name]["pH"]

        N_scale = calc_scale(nitrogen,N_normal,0.05)
        P_scale = calc_scale(phosphorous,P_normal,0.05)
        K_scale = calc_scale(pottasium,K_normal,0.05)
        pH_scale = calc_scale(ph,pH_normal,0.2)
        
        result ={
            "crop" : crop_name,
            "N": nitrogen,
            "P": phosphorous,
            "K" : pottasium,
            "Temp" : temperature,
            "Humidity" : humidity,
            "pH" : ph,
            "N_normal" : float(N_normal),
            "P_normal" : float(P_normal),
            "K_normal" : float(K_normal),
            "pH_normal" :float(pH_normal),
            "N_scale" : N_scale,
            "P_scale" : P_scale,
            "K_scale" : K_scale,
            "pH_scale" : pH_scale
        }
        return result
    else:
        return {"STATUS":"ERROR"}

def predict_fertilizer_service(cropname,nitrogen,phosphorous,pottasium):
    crop_name = cropname.lower()
    soil_N = nitrogen
    soil_P = phosphorous
    soil_K = pottasium
    
    df = pd.read_csv('app/Data/fertilizer.csv')

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
    K_scale = calc_scale(pottasium,K_normal,0.05)
    # Create the response dictionary
    data = {
        'soil_N': convert_to_python_type(soil_N),
        'soil_P': convert_to_python_type(soil_P),
        'soil_K': convert_to_python_type(soil_K),
        'N_normal': convert_to_python_type(N_normal),
        'P_normal': convert_to_python_type(P_normal),
        'K_normal': convert_to_python_type(K_normal),
        'soil_N_level': soil_N_level,
        'soil_P_level': soil_P_level,
        'soil_K_level': soil_K_level,
        'N_scale': N_scale,
        'P_scale': P_scale,
        'K_scale': K_scale
    }

    return data

#print(predict_fertilizer_service("watermelon",80,7,10))