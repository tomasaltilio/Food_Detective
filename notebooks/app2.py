import streamlit as st
from PIL import Image
import requests
import h5py
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
import time
import sys
import pandas as pd
import json
import tensorflow as tf

CSS = """
    h1 {
    color: black;
    }
    body {
    background-image: url(https://raw.githubusercontent.com/tomasaltilio/Food_Detective/test_movile/notebooks/bacground2.png);
    background-size: cover;
    }
    """
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

categories = [b'apple_pie',
              b'baby_back_ribs',
              b'baklava',
              b'beef_carpaccio',
              b'beef_tartare',
              b'beet_salad',
              b'beignets',
              b'bibimbap',
              b'bread_pudding',
              b'breakfast_burrito',
              b'bruschetta',
              b'caesar_salad',
              b'cannoli',
              b'caprese_salad',
              b'carrot_cake',
              b'ceviche',
              b'cheese_plate',
              b'cheesecake',
              b'chicken_curry',
              b'chicken_quesadilla',
              b'chicken_wings',
              b'chocolate_cake',
              b'chocolate_mousse',
              b'churros',
              b'clam_chowder',
              b'club_sandwich',
              b'crab_cakes',
              b'creme_brulee',
              b'croque_madame',
              b'cup_cakes',
              b'deviled_eggs',
              b'donuts',
              b'dumplings',
              b'edamame',
              b'eggs_benedict',
              b'escargots',
              b'falafel',
              b'filet_mignon',
              b'fish_and_chips',
              b'foie_gras',
              b'french_fries',
              b'french_onion_soup',
              b'french_toast',
              b'fried_calamari',
              b'fried_rice',
              b'frozen_yogurt',
              b'garlic_bread',
              b'gnocchi',
              b'greek_salad',
              b'grilled_cheese_sandwich',
              b'grilled_salmon',
              b'guacamole',
              b'gyoza',
              b'hamburger',
              b'hot_and_sour_soup',
              b'hot_dog',
              b'huevos_rancheros',
              b'hummus',
              b'ice_cream',
              b'lasagna',
              b'lobster_bisque',
              b'lobster_roll_sandwich',
              b'macaroni_and_cheese',
              b'macarons',
              b'miso_soup',
              b'mussels',
              b'nachos',
              b'omelette',
              b'onion_rings',
              b'oysters',
              b'pad_thai',
              b'paella',
              b'pancakes',
              b'panna_cotta',
              b'peking_duck',
              b'pho',
              b'pizza',
              b'pork_chop',
              b'poutine',
              b'prime_rib',
              b'pulled_pork_sandwich',
              b'ramen',
              b'ravioli',
              b'red_velvet_cake',
              b'risotto',
              b'samosa',
              b'sashimi',
              b'scallops',
              b'seaweed_salad',
              b'shrimp_and_grits',
              b'spaghetti_bolognese',
              b'spaghetti_carbonara',
              b'spring_rolls',
              b'steak',
              b'strawberry_shortcake',
              b'sushi',
              b'tacos',
              b'takoyaki',
              b'tiramisu',
              b'tuna_tartare',
              b'waffles']

def import_model(img):
  interpreter = tf.lite.Interpreter(model_path='/mnt/c/Users/dani_/Downloads/model-export-tflite-20210316T180116Z-001/model-export-tflite/model.tflite')
  cats = pd.read_csv('/mnt/c/Users/dani_/Downloads/model-export-tflite-20210316T180116Z-001/model-export-tflite/dict.txt', header=None)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  floating_model = input_details[0]['dtype'] == np.float32
  input_data = np.expand_dims(img, axis=0)
  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std
  interpreter.set_tensor(input_details[0]['index'], input_data)
  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)
  categories =  cats.values.tolist()
  cat_name = categories[np.argmax(results)]
  return cat_name[0]

def get_api_info(category_name_sample):
    """Function to get the api information"""
    api_url = 'https://api.calorieninjas.com/v1/nutrition?query='
    query = '100g {}'.format(category_name_sample)
    response = requests.get(api_url + query, headers={'X-Api-Key': 'Xy0klDLLr8umMOU3GBTv8g==ixVHYHMXsQA3CJT7'})
    if response.status_code == requests.codes.ok:
        print(response.text)
        return response.text
    else:
        print("Error:", response.status_code, response.text)

#    print(response.text)

def convert_data(api_info):
    df = pd.DataFrame.from_dict(api_info['items'][0], orient='index').T
    df = df[['name','sugar_g', 'fiber_g', 'serving_size_g', 'sodium_mg',
             'potassium_mg', 'fat_saturated_g', 'fat_total_g', 'calories',
             'cholesterol_mg', 'protein_g', 'carbohydrates_total_g']]
    df.columns = ['Name','Sugar', 'Fiber', 'Serving Size', 'Sodium',
                  'Potassium', 'Fat Saturated', 'Fat Total', 'Calories',
                'Cholesterol', 'Protein', 'TotalCarbohydrates']
    return df

def add_statement(df):
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

def warnings_men(df):
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0]>(36/4):
        value = round(df["sugar_g"][0],ndigits=2)
        ":warning: Attention! You are having more than the sugar levels recommended per meal "
    if df['fiber_g'][0]>(38/4):
        value = round(df["fiber_g"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the fiber levels recommended per meal "
    if df['sodium_mg'][0]>(2.3/4):
        value = round(df["sodium_mg"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the sodium levels recommended per meal "
    if df['potassium_mg'][0]>(4.7/4):
        value = round(df["potassium_mg"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the potassium levels recommended per meal "
    if df['fat_saturated_g'][0]>(22/4):
        value = round(df["fat_saturated_g"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the saturated fat levels recommended per meal "
    if df['fat_total_g'][0]>(77/4):
        value = round(df["fat_total_g"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the total fat levels recommended per meal "
    if df['calories'][0]>(2500/4):
        value = round(df["calories"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the calories levels recommended per meal "
    if df['cholesterol_mg'][0]>(0.3/4):
        value = round(df["cholesterol_mg"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the cholesterol level recommended per meal "
#    if df['protein_g'][0]>(0.8*weight/2):
#        value = round(df["protein_g"][0]/0.25,ndigits=2)
#        st.error(f"Attention! You are having more than the protein levels recommended per meal ")
    if df['carbohydrates_total_g'][0]>(300/4):
        value = round(df["carbohydrates_total_g"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the carbohydrate levels recommended per meal "
    else:
        'The rest of the levels are just ok :muscle:!'

def warnings_women(df):
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0]>(25/4):
        value = round(df["sugar_g"][0],ndigits=2)
        ":warning: Attention! You are having more than the sugar levels recommended per meal "
    if df['fiber_g'][0]>25/4:
        value = round(df["fiber_g"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the fiber levels recommended per meal "
    if df['sodium_mg'][0]>(2.3/4):
        value = round(df["sodium_mg"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the sodium levels recommended per meal "
    if df['potassium_mg'][0]>(4.7/4):
        value = round(df["potassium_mg"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the potassium levels recommended per meal "
    if df['fat_saturated_g'][0]>(22/4):
        value = round(df["fat_saturated_g"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the saturated fat levels recommended per meal "
    if df['fat_total_g'][0]>(77/4):
        value = round(df["fat_total_g"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the total fat levels recommended per meal "
    if df['calories'][0]>(2000/4):
        value = round(df["calories"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the calories levels recommended per meal "
    if df['cholesterol_mg'][0]>(0.3/4):
        value = round(df["cholesterol_mg"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the cholesterol level recommended per meal "
#    if df['protein_g'][0]>():
#        value = round(df["protein_g"][0]/0.25,ndigits=2)
#        st.error(f"Attention! You are having more than the protein levels recommended per meal ")
    if df['carbohydrates_total_g'][0]>(300/4):
        value = round(df["carbohydrates_total_g"][0]/0.25,ndigits=2)
        ":warning: Attention! You are having more than the carbohydrate levels recommended per meal "
    else:
        'The rest of the levels are just ok :muscle:!'

#url = 'https://res.cloudinary.com/sanitarium/image/fetch/q_auto/https://www.sanitarium.com.au/getmedia%2Fae51f174-984f-4a70-ad3d-3f6b517b6da1%2Ffruits-vegetables-healthy-fats.jpg%3Fwidth%3D1180%26height%3D524%26ext%3D.jpg'
#
#st.image(url, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
st.markdown('''# Hi! Welcome to Food Detective :green_salad: :mag: :eyes:''')
'Upload a photo of your meal to know about its nutritional information!'

with st.beta_expander("Search image..."):
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        imagen = Image.open(uploaded_file)
        st.image(imagen, use_column_width=True)
        new_width  = 224
        new_height = 224
        imagen = imagen.resize((new_width, new_height), Image.ANTIALIAS)
        st.write("")
        #linea para hace predict
        with st.spinner('Please wait! We are inspecting your meal...'):
            time.sleep(5)
            category_name_sample = import_model(imagen)
            api_info_text = get_api_info(category_name_sample)
            df_api = json.loads(api_info_text)
            api_info = convert_data(df_api)
            api_info_transformed = add_statement(api_info)
        st.success(':white_check_mark: Done! You are having...')
        f'**{category_name_sample}**!'


        'To know about its nutritional information please complete your gender first:'
        gender = st.radio('Gender:',('Select','Male', 'Female'))
            
        if gender == 'Female':
            api_info_transformed
            warnings_women(df_api)
        
        if gender == 'Male':
            api_info_transformed
            warnings_men(df_api)



