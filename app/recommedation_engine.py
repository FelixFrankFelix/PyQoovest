from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key)

def factor_crop_rec(factor, factor_value, factor_normal, crop_name):
    exception_status = "NO"
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a recommender for rural farmers. Only use blocks of text in 100 words strictly and make responses dynamic. The response should be in 2 paragraphs: the first paragraph should address general practice, and the second paragraph should offer rural alternatives to help farmers. You may include solutions or resources native to Kenya with Kenyan traditional names. If a Kenyan traditional name is used, describe it briefly."},
                {"role": "user", "content": f"""Provide a recommendation to improve/maintain the {factor} level of soil for planting {crop_name}.
                The soil has {factor_value} of {factor} and the average {factor} for {crop_name} is {factor_normal} or within that range."""}
            ]
        )
        return completion.choices[0].message.content,exception_status
    except Exception as e:
        exception_status="YES"
        return e,exception_status

def factor_fert_rec(N,P,K,N_level,P_level,K_level,N_normal,P_normal,K_normal,crop_name):
    exception_status = "NO"
    try:
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": "You are a recommender for rural farmers. only use block of text in 100 words strictly and make responses dynamic, The response should be in 3 paragraphs, first paragraph should speak to general practice and second paragraph should speak to rural alternative to help farmers(you could add solutions/resources native to Kenya, with Kenyan traditional names, if you add Kenyan traditional name, describe it briefly),The third paragraph should speak to Defiency Consequence/Effect"},
            {"role": "user", "content": f"""Provide Recommendation for fertilizer to add to soil of profile, Nitrogen: {N}, Phosphorus {P} and Potassium {K} to plant {crop_name}
                                            To plant {crop_name}, Nitrogen Level of your soil is {N_level} for optimal value:{N_normal},
                                            Phosphorus Level of your soil is {P_level} for optimal value:{P_normal},
                                            Potassium Level of your soil is {K_level} for optimal value:{K_normal}.
                                            For this soil profile, give in bullet points effect {N_level} level Nitrogen, {P_level} level Phosphorus and {K_level} level Potassium"""},
    ]

    )
        return completion.choices[0].message.content,exception_status
    except Exception as e:
        exception_status="YES"
        return e,exception_status

def factor_disease_rec(crop_name,disease_name):
    exception_status = "NO"
    try:
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": "You are a recommender for rural farmers. only use block of text in 100 words strictly and make responses dynamic, The response should be in 2 paragraphs, first paragraph should speak to Causes of the disease and the second paragraph should speak to remedies, you should add kenyan local remedies, if you add Kenyan traditional name, describe it briefly)"},
            {"role": "user", "content": f"""Provide Recommendation for disease diagnosis for {crop_name} having {disease_name},the recommedations should be tailored to the crop and if the crop condition({disease_name}) is healty, provide recommendation for maintainance"""},
    ]

    )
        return completion.choices[0].message.content,exception_status
    except Exception as e:
        exception_status="YES"
        return e,exception_status


#print(factor_fert_rec(20,20,20,"Low","High","Normal",30,10,21,"corn"))