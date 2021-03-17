# Functions used in app

# Library imports
import streamlit as st
from tensorflow import lite
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras import models
import tensorflow.keras.backend as K
from Food_Detective.params import categories
import numpy as np
import pandas as pd
import requests
from PIL import Image
import os


@st.cache(allow_output_mutation=True, show_spinner=False)
def download_model():
    """Function that downloads the model"""
    path = os.path.join(os.getcwd(), 'model-export-tflite/model.tflite')
    interpreter = lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def preprocessing_func(image, interpreter):
    '''Function that preprocesses the input'''
    input_details = interpreter.get_input_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = image.resize((width, height))
    input_data = np.expand_dims(img, axis=0)
    return input_data


def predict_category(interpreter, input_data):
    """Function that predicts the category of the image"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    category_name_sample = categories[np.argmax(results)][0]
    category_name_sample = category_name_sample.capitalize().replace("_", " ")
    return category_name_sample


def get_api_info(category_name_sample):
    """Function to get the api information"""
    api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
    query = '100g {}'.format(category_name_sample)
    response = requests.get(
        api_url + query, headers={'X-Api-Key': 'Xy0klDLLr8umMOU3GBTv8g==ixVHYHMXsQA3CJT7'})
    if response.status_code == requests.codes.ok:
        if response.text == '{"items": []}':
            return 'Error'
        return response.text
    else:
        print("Error:", response.status_code, response.text)


def convert_data(api_info):
    '''Function that converts the dictionary provided by the api and turns it into a table'''
    df = pd.DataFrame.from_dict(api_info['items'][0], orient='index').T
    df = df[['name', 'sugar_g', 'fiber_g', 'serving_size_g', 'sodium_mg',
             'potassium_mg', 'fat_saturated_g', 'fat_total_g', 'calories',
             'cholesterol_mg', 'protein_g', 'carbohydrates_total_g']]
    df.columns = ['Name', 'Sugar', 'Fiber', 'Serving Size', 'Sodium',
                  'Potassium', 'Fat Saturated', 'Fat Total', 'Calories',
                  'Cholesterol', 'Protein', 'TotalCarbohydrates']
    return df


def add_statement(df):
    '''Function that adds labels to the data provided by the api'''
    Sugar = df['Sugar'][0]
    Fiber = df['Fiber'][0]
    Serving_Size = df['Serving Size'][0]
    Sodium = df['Sodium'][0]
    Potassium = df['Potassium'][0]
    Fat_Saturated = df['Fat Saturated'][0]
    Fat_Total = df['Fat Total'][0]
    Cholesterol = df['Cholesterol'][0]
    Protein = df['Protein'][0]
    TotalCarbohydrates = df['TotalCarbohydrates'][0]
    df['Sugar'] = f'{Sugar}g'
    df['Fiber'] = f'{Fiber}g'
    df['Serving Size'] = f'{Serving_Size}g'
    df['Sodium'] = f'{Sodium}mg'
    df['Potassium'] = f'{Potassium}mg'
    df['Fat Saturated'] = f'{Fat_Saturated}g'
    df['Fat Total'] = f'{Fat_Total}g'
    df['Cholesterol'] = f'{Cholesterol}g'
    df['Protein'] = f'{Protein}g'
    df['TotalCarbohydrates'] = f'{TotalCarbohydrates}g'
    df_t = df.T
    df_t.columns = ['']
    return df_t



def warnings_men_young(df):
    '''Displays health warnings for men under 50 years'''
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0] > (38/4):
        value_sugar = round(df["sugar_g"][0], ndigits=2)
        st.write(
            f':warning: Sugar: You are having more than the sugar levels recommended per meal. You should take 38 grams  per day, but careful, your meal already has {value_sugar} grams of sugar!')
    if df['fiber_g'][0] > (35/4):
        value_fiber = round(df["fiber_g"], ndigits=2)
        st.write(
            f':warning: Fiber: You are having more than the fiber  levels recommended per meal. You should take 35 grams  per day, but careful, your meal already has {value_fiber} grams of fat!')
    if df['sodium_mg'][0] > (2.3/4):
        value_sodium = round(df["sodium_mg"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Sodium: You are having more than the saturated sodium levels recommended per meal. You should take 2.3 grams per day, but careful, your meal already has {value_sodium} grams of sodium! ')
    if df['potassium_mg'][0] > (4.7/4):
        value_pot = round(df["potassium_mg"][0]/0.25, ndigits=2)
        st.write(
            f":warning: Potassium: You are having more than the potassium levels recommended per meal. You should take 4.7 grams per day, but careful, your meal already has {value_pot} grams of potassium!")
    if df['fat_saturated_g'][0] > (27/4):
        value_satfat = round(df["fat_saturated_g"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Attention! You are having more than the saturated fat levels recommended per meal. You should take 27 grams per day, but careful, your meal already has {value_satfat} grams of saturated fat!')
    if df['fat_total_g'][0] > (88/4):
        value_fat = round(df["fat_total_g"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Fat: You are having more than the total fat levels recommended per meal. You should take 88 grams  per day, but careful, your meal already has {value_fat} grams of fat! ')
    if df['calories'][0] > (2700/4):
        value = round(df["calories"][0], ndigits=2)
        st.write(f':warning: Calories: You are having more than the calories levels recommended per meal. You should take 675 calories per meal if you are making 4 meals per day. "')
    if df['cholesterol_mg'][0] > (0.3/4):
        value_chol = round(df["cholesterol_mg"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Cholesterol: You are having more than the cholesterol levels recommended per meal. You should take 0.3 grams  per day, but careful, your meal already has {value_chol} grams of cholesterol! ')
    if df['protein_g'][0] > (63/4):
        value_protein = round(df["protein_g"][0], ndigits=2)
        st.write(
            f':warning:  Protein: You are having more than the protein levels recommended per meal. You should take 63g  per day, but your meal already has {value_protein} grams of protein.')
    if df['carbohydrates_total_g'][0] > (410/4):
        value_carbs = round(df["carbohydrates_total_g"][0], ndigits=2)
        st.write(
            f':warning: Carbohydrates: You are having more than the carbohydrate levels recommended per meal. You should take 410g  per day, but your meal already has {value_carbs} grams of protein.')
    else:
        st.write('The rest of the levels are just ok :muscle:!')


def warnings_men_old(df):
    '''Displays health warnings for men over 50 years'''
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0] > (38/4):
        value_sugar = round(df["sugar_g"][0], ndigits=2)
        st.write(
            f':warning: Sugar: You are having more than the sugar levels recommended per meal. You should take 38 grams  per day, but careful, your meal already has {value_sugar} grams of sugar!')
    if df['fiber_g'][0] > (35/4):
        value_fiber = round(df["fiber_g"], ndigits=2)
        st.write(
            f':warning: Fiber: You are having more than the fiber  levels recommended per meal. You should take 35 grams  per day, but careful, your meal already has {value_fiber} grams of fat!')
    if df['sodium_mg'][0] > (1.5/4):
        value_sodium = round(df["sodium_mg"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Sodium: You are having more than the saturated sodium levels recommended per meal. You should take 2.3 grams per day, but careful, your meal already has {value_sodium} grams of sodium! ')
    if df['potassium_mg'][0] > (4.7/4):
        value = round(df["potassium_mg"][0]/0.25, ndigits=2)
        st.write(
            ":warning: Potassium: You are having more than the potassium levels recommended per meal ")
    if df['fat_saturated_g'][0] > (25/4):
        value_satfat = round(df["fat_saturated_g"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Saturated fat: You are having more than the saturated fat levels recommended per meal. You should take 27 grams per day, but careful, your meal already has {value_satfat} grams of saturated fat!')
    if df['fat_total_g'][0] > (83/4):
        value_fat = round(df["fat_total_g"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Total fat: You are having more than the total fat levels recommended per meal. You should take 88 grams  per day, but careful, your meal already has {value_fat} grams of fat! ')
    if df['calories'][0] > (2500/4):
        value = round(df["calories"][0], ndigits=2)
        st.write(f':warning: Calories: You are having more than the calories levels recommended per meal. You should take 675 calories per meal if you are making 4 meals per day. "')
    if df['cholesterol_mg'][0] > (0.3/4):
        value_chol = round(df["cholesterol_mg"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Cholesterol: You are having more than the cholesterol levels recommended per meal. You should take 0.3 grams  per day, but careful, your meal already has {value_chol} grams of cholesterol! ')
    if df['protein_g'][0] > (60/4):
        value_protein = round(df["protein_g"][0], ndigits=2)
        st.write(
            f':warning:  Proteins:  You are having more than the protein levels recommended per meal. You should take 63g  per day, but your meal already has {value_protein} grams of protein.')
    if df['carbohydrates_total_g'][0] > (375/4):
        value_carbs = round(df["carbohydrates_total_g"][0], ndigits=2)
        st.write(
            f':warning: Carbohydrates: You are having more than the carbohydrate levels recommended per meal. You should take 410g  per day, but your meal already has {value_carbs} grams of protein.')
    else:
        st.write('The rest of the levels are just ok :muscle:!')


def warnings_women_old(df):
    '''Displays health warnings for women over 50 years'''
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0] > (38/4):
        value_sugar = round(df["sugar_g"][0], ndigits=2)
        st.write(
            f':warning: Sugar: You are having more than the sugar levels recommended per meal. You should take 38 grams  per day, but careful, your meal already has {value_sugar} grams of sugar!')
    if df['fiber_g'][0] > (35/4):
        value_fiber = round(df["fiber_g"], ndigits=2)
        st.write(
            f':warning: Fiber: You are having more than the fiber  levels recommended per meal. You should take 35 grams  per day, but careful, your meal already has {value_fiber} grams of fat!')
    if df['sodium_mg'][0] > (1.5/4):
        value_sodium = round(df["sodium_mg"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Sodium: You are having more than the saturated sodium levels recommended per meal. You should take 2.3 grams per day, but careful, your meal already has {value_sodium} grams of sodium! ')
    if df['potassium_mg'][0] > (4.7/4):
        value = round(df["potassium_mg"][0]/0.25, ndigits=2)
        st.write(
            ":warning: Potassium: You are having more than the potassium levels recommended per meal ")
    if df['fat_saturated_g'][0] > (20/4):
        value_satfat = round(df["fat_saturated_g"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Saturated fat: You are having more than the saturated fat levels recommended per meal. You should take 27 grams per day, but careful, your meal already has {value_satfat} grams of saturated fat!')
    if df['fat_total_g'][0] > (65/4):
        value_fat = round(df["fat_total_g"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Total fat: You are having more than the total fat levels recommended per meal. You should take 88 grams  per day, but careful, your meal already has {value_fat} grams of fat! ')
    if df['calories'][0] > (2000/4):
        value = round(df["calories"][0], ndigits=2)
        st.write(f':warning: Calories: You are having more than the calories levels recommended per meal. You should take 675 calories per meal if you are making 4 meals per day. "')
    if df['cholesterol_mg'][0] > (0.3/4):
        value_chol = round(df["cholesterol_mg"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Cholesterol: You are having more than the cholesterol levels recommended per meal. You should take 0.3 grams  per day, but careful, your meal already has {value_chol} grams of cholesterol! ')
    if df['protein_g'][0] > (50/4):
        value_protein = round(df["protein_g"][0], ndigits=2)
        st.write(
            f':warning:  Proteins: You are having more than the protein levels recommended per meal. You should take 63g  per day, but your meal already has {value_protein} grams of protein.')
    if df['carbohydrates_total_g'][0] > (304/4):
        value_carbs = round(df["carbohydrates_total_g"][0], ndigits=2)
        st.write(
            f':warning: Carbohydrates: You are having more than the carbohydrate levels recommended per meal. You should take 410g  per day, but your meal already has {value_carbs} grams of protein.')
    else:
        st.write('The rest of the levels are just ok :muscle:!')


def warnings_women_young(df):
    '''Displays health warnings for women under 50 years'''
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0] > (38/4):
        value_sugar = round(df["sugar_g"][0], ndigits=2)
        st.write(
            f':warning: Attention! You are having more than the sugar levels recommended per meal. You should take 38 grams  per day, but careful, your meal already has {value_sugar} grams of sugar!')
    if df['fiber_g'][0] > (35/4):
        value_fiber = round(df["fiber_g"], ndigits=2)
        st.write(
            f':warning: Attention! You are having more than the fiber  levels recommended per meal. You should take 35 grams  per day, but careful, your meal already has {value_fiber} grams of fat!')
    if df['sodium_mg'][0] > (2.3/4):
        value_sodium = round(df["sodium_mg"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Sodium: You are having more than the saturated sodium levels recommended per meal. You should take 2.3 grams per day, but careful, your meal already has {value_sodium} grams of sodium! ')
    if df['potassium_mg'][0] > (4.7/4):
        value = round(df["potassium_mg"][0]/0.25, ndigits=2)
        st.write(
            ":warning: Potassium: You are having more than the potassium levels recommended per meal ")
    if df['fat_saturated_g'][0] > (20/4):
        value_satfat = round(df["fat_saturated_g"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Saturated fat: You are having more than the saturated fat levels recommended per meal. You should take 27 grams per day, but careful, your meal already has {value_satfat} grams of saturated fat!')
    if df['fat_total_g'][0] > (65/4):
        value_fat = round(df["fat_total_g"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Total fat: You are having more than the total fat levels recommended per meal. You should take 88 grams  per day, but careful, your meal already has {value_fat} grams of fat! ')
    if df['calories'][0] > (2000/4):
        value = round(df["calories"][0], ndigits=2)
        st.write(f':warning: Calories: You are having more than the calories levels recommended per meal. You should take 675 calories per meal if you are making 4 meals per day. "')
    if df['cholesterol_mg'][0] > (0.3/4):
        value_chol = round(df["cholesterol_mg"][0]/0.25, ndigits=2)
        st.write(
            f':warning: Cholesterol: You are having more than the cholesterol levels recommended per meal. You should take 0.3 grams  per day, but careful, your meal already has {value_chol} grams of cholesterol! ')
    if df['protein_g'][0] > (50/4):
        value_protein = round(df["protein_g"][0], ndigits=2)
        st.write(
            f':warning:  Protein: You are having more than the protein levels recommended per meal. You should take 63g  per day, but your meal already has {value_protein} grams of protein.')
    if df['carbohydrates_total_g'][0] > (304/4):
        value_carbs = round(df["carbohydrates_total_g"][0], ndigits=2)
        st.write(
            f':warning: Carbohydrates: You are having more than the carbohydrate levels recommended per meal. You should take 410g  per day, but your meal already has {value_carbs} grams of protein.')
    else:
        st.write('The rest of the levels are just ok :muscle:!')
