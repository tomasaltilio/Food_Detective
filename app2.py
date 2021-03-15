import streamlit as st
from PIL import Image
import requests
import h5py
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
import tensorflow.keras.backend as K
import time
import sys
import pandas as pd
import json


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

def preprocessing_func(image):
    '''Function that preprocesses the input'''
    preproc_img = img_to_array(image)
    preproc_img = np.expand_dims(preproc_img, axis = 0)
    preproc_img = preprocess_input(preproc_img)
    return preproc_img

@st.cache(allow_output_mutation=True, show_spinner=False)
def download_model():
    """Function that downloads the model"""
    path = 'EfficientNet_acc39.h5'
    model = models.load_model(path)
    model.make_predict_function()
    model.summary()
    return model

def predict_category(model,image):
    """Function that predicts the category of the image"""
    prediction = model.predict(image)
    category_sample = np.argmax(prediction)
    category_name_sample = categories[category_sample]
    category_name_sample = category_name_sample.decode('UTF-8').capitalize().replace("_", " ")
    return category_name_sample

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

def warnings_men_young(df):
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0]>(38/4):
        value_sugar = round(df["sugar_g"][0],ndigits=2)
        f'warning: Sugar: You are having more than the sugar levels recommended per meal. You should take 38 grams  per day, but careful, your meal already has {value_sugar} grams of sugar!'
    if df['fiber_g'][0]>(35/4):
        value_fiber = round(df["fiber_g"],ndigits=2)
        f'warning: Fiber: You are having more than the fiber  levels recommended per meal. You should take 35 grams  per day, but careful, your meal already has {value_fiber} grams of fat!'
    if df['sodium_mg'][0]>(2.3/4):
        value_sodium = round(df["sodium_mg"][0]/0.25,ndigits=2)
        f':warning: Sodium: You are having more than the saturated sodium levels recommended per meal. You should take 2.3 grams per day, but careful, your meal already has {value_sodium} grams of sodium! '
    if df['potassium_mg'][0]>(4.7/4):
        value = round(df["potassium_mg"][0]/0.25,ndigits=2)
        ":warning: Potassium: You are having more than the potassium levels recommended per meal "
    if df['fat_saturated_g'][0]>(27/4):
        value_satfat = round(df["fat_saturated_g"][0]/0.25,ndigits=2)
        f':warning: Attention! You are having more than the saturated fat levels recommended per meal. You should take 27 grams per day, but careful, your meal already has {value_satfat} grams of saturated fat!'
    if df['fat_total_g'][0]>(88/4):
        value_fat = round(df["fat_total_g"][0]/0.25,ndigits=2)
        f':warning: Fat: You are having more than the total fat levels recommended per meal. You should take 88 grams  per day, but careful, your meal already has {value_fat} grams of fat! '
    if df['calories'][0]>(2700/4):
        value = round(df["calories"][0],ndigits=2)
        f':warning: Calories: You are having more than the calories levels recommended per meal. You should take 675 calories per meal if you are making 4 meals per day. "'
    if df['cholesterol_mg'][0]>(0.3/4):
        value_chol = round(df["cholesterol_mg"][0]/0.25,ndigits=2)
        f':warning: Cholesterol: You are having more than the cholesterol levels recommended per meal. You should take 0.3 grams  per day, but careful, your meal already has {value_chol} grams of cholesterol! '
    if df['protein_g'][0]>(63/4):
        value_protein = round(df["protein_g"][0],ndigits=2)
        f':warning:  Protein: You are having more than the protein levels recommended per meal. You should take 63g  per day, but your meal already has {value_protein} grams of protein.'
    if df['carbohydrates_total_g'][0]>(410/4):
        value_carbs = round(df["carbohydrates_total_g"][0],ndigits=2)
        f':warning: Carbohydrates: You are having more than the carbohydrate levels recommended per meal. You should take 410g  per day, but your meal already has {value_carbs} grams of protein.'
    else:
        'The rest of the levels are just ok :muscle:!'

def warnings_men_old(df):
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0]>(38/4):
        value_sugar = round(df["sugar_g"][0],ndigits=2)
        f'warning: Sugar: You are having more than the sugar levels recommended per meal. You should take 38 grams  per day, but careful, your meal already has {value_sugar} grams of sugar!'
    if df['fiber_g'][0]>(35/4):
        value_fiber = round(df["fiber_g"],ndigits=2)
        f'warning: Fiber: You are having more than the fiber  levels recommended per meal. You should take 35 grams  per day, but careful, your meal already has {value_fiber} grams of fat!'
    if df['sodium_mg'][0]>(1.5/4):
        value_sodium = round(df["sodium_mg"][0]/0.25,ndigits=2)
        f':warning: Sodium: You are having more than the saturated sodium levels recommended per meal. You should take 2.3 grams per day, but careful, your meal already has {value_sodium} grams of sodium! '
    if df['potassium_mg'][0]>(4.7/4):
        value = round(df["potassium_mg"][0]/0.25,ndigits=2)
        ":warning: Potassium: You are having more than the potassium levels recommended per meal "
    if df['fat_saturated_g'][0]>(25/4):
        value_satfat = round(df["fat_saturated_g"][0]/0.25,ndigits=2)
        f':warning: Saturated fat: You are having more than the saturated fat levels recommended per meal. You should take 27 grams per day, but careful, your meal already has {value_satfat} grams of saturated fat!'
    if df['fat_total_g'][0]>(83/4):
        value_fat = round(df["fat_total_g"][0]/0.25,ndigits=2)
        f':warning: Total fat: You are having more than the total fat levels recommended per meal. You should take 88 grams  per day, but careful, your meal already has {value_fat} grams of fat! '
    if df['calories'][0]>(2500/4):
        value = round(df["calories"][0],ndigits=2)
        f':warning: Calories: You are having more than the calories levels recommended per meal. You should take 675 calories per meal if you are making 4 meals per day. "'
    if df['cholesterol_mg'][0]>(0.3/4):
        value_chol = round(df["cholesterol_mg"][0]/0.25,ndigits=2)
        f':warning: Cholesterol: You are having more than the cholesterol levels recommended per meal. You should take 0.3 grams  per day, but careful, your meal already has {value_chol} grams of cholesterol! '
    if df['protein_g'][0]>(60/4):
        value_protein = round(df["protein_g"][0],ndigits=2)
        f':warning:  Proteins:  You are having more than the protein levels recommended per meal. You should take 63g  per day, but your meal already has {value_protein} grams of protein.'
    if df['carbohydrates_total_g'][0]>(375/4):
        value_carbs = round(df["carbohydrates_total_g"][0],ndigits=2)
        f':warning: Carbohydrates: You are having more than the carbohydrate levels recommended per meal. You should take 410g  per day, but your meal already has {value_carbs} grams of protein.'
    else:
        'The rest of the levels are just ok :muscle:!'

def warnings_women_old(df):
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0]>(38/4):
        value_sugar = round(df["sugar_g"][0],ndigits=2)
        f'warning: Sugar: You are having more than the sugar levels recommended per meal. You should take 38 grams  per day, but careful, your meal already has {value_sugar} grams of sugar!'
    if df['fiber_g'][0]>(35/4):
        value_fiber = round(df["fiber_g"],ndigits=2)
        f'warning: Fiber: You are having more than the fiber  levels recommended per meal. You should take 35 grams  per day, but careful, your meal already has {value_fiber} grams of fat!'
    if df['sodium_mg'][0]>(1.5/4):
        value_sodium = round(df["sodium_mg"][0]/0.25,ndigits=2)
        f':warning: Sodium: You are having more than the saturated sodium levels recommended per meal. You should take 2.3 grams per day, but careful, your meal already has {value_sodium} grams of sodium! '
    if df['potassium_mg'][0]>(4.7/4):
        value = round(df["potassium_mg"][0]/0.25,ndigits=2)
        ":warning: Potassium: You are having more than the potassium levels recommended per meal "
    if df['fat_saturated_g'][0]>(20/4):
        value_satfat = round(df["fat_saturated_g"][0]/0.25,ndigits=2)
        f':warning: Saturated fat: You are having more than the saturated fat levels recommended per meal. You should take 27 grams per day, but careful, your meal already has {value_satfat} grams of saturated fat!'
    if df['fat_total_g'][0]>(65/4):
        value_fat = round(df["fat_total_g"][0]/0.25,ndigits=2)
        f':warning: Total fat: You are having more than the total fat levels recommended per meal. You should take 88 grams  per day, but careful, your meal already has {value_fat} grams of fat! '
    if df['calories'][0]>(2000/4):
        value = round(df["calories"][0],ndigits=2)
        f':warning: Calories: You are having more than the calories levels recommended per meal. You should take 675 calories per meal if you are making 4 meals per day. "'
    if df['cholesterol_mg'][0]>(0.3/4):
        value_chol = round(df["cholesterol_mg"][0]/0.25,ndigits=2)
        f':warning: Cholesterol: You are having more than the cholesterol levels recommended per meal. You should take 0.3 grams  per day, but careful, your meal already has {value_chol} grams of cholesterol! '
    if df['protein_g'][0]>(50/4):
        value_protein = round(df["protein_g"][0],ndigits=2)
        f':warning:  Proteins: You are having more than the protein levels recommended per meal. You should take 63g  per day, but your meal already has {value_protein} grams of protein.'
    if df['carbohydrates_total_g'][0]>(304/4):
        value_carbs = round(df["carbohydrates_total_g"][0],ndigits=2)
        f':warning: Carbohydrates: You are having more than the carbohydrate levels recommended per meal. You should take 410g  per day, but your meal already has {value_carbs} grams of protein.'
    else:
        'The rest of the levels are just ok :muscle:!'


def warnings_women_young(df):
    df = pd.DataFrame.from_dict(df['items'][0], orient='index').T
    if df['sugar_g'][0]>(38/4):
        value_sugar = round(df["sugar_g"][0],ndigits=2)
        f'warning: Attention! You are having more than the sugar levels recommended per meal. You should take 38 grams  per day, but careful, your meal already has {value_sugar} grams of sugar!'
    if df['fiber_g'][0]>(35/4):
        value_fiber = round(df["fiber_g"],ndigits=2)
        f'warning: Attention! You are having more than the fiber  levels recommended per meal. You should take 35 grams  per day, but careful, your meal already has {value_fiber} grams of fat!'
    if df['sodium_mg'][0]>(2.3/4):
        value_sodium = round(df["sodium_mg"][0]/0.25,ndigits=2)
        f':warning: Sodium: You are having more than the saturated sodium levels recommended per meal. You should take 2.3 grams per day, but careful, your meal already has {value_sodium} grams of sodium! '
    if df['potassium_mg'][0]>(4.7/4):
        value = round(df["potassium_mg"][0]/0.25,ndigits=2)
        ":warning: Potassium: You are having more than the potassium levels recommended per meal "
    if df['fat_saturated_g'][0]>(20/4):
        value_satfat = round(df["fat_saturated_g"][0]/0.25,ndigits=2)
        f':warning: Saturated fat: You are having more than the saturated fat levels recommended per meal. You should take 27 grams per day, but careful, your meal already has {value_satfat} grams of saturated fat!'
    if df['fat_total_g'][0]>(65/4):
        value_fat = round(df["fat_total_g"][0]/0.25,ndigits=2)
        f':warning: Total fat: You are having more than the total fat levels recommended per meal. You should take 88 grams  per day, but careful, your meal already has {value_fat} grams of fat! '
    if df['calories'][0]>(2000/4):
        value = round(df["calories"][0],ndigits=2)
        f':warning: Calories: You are having more than the calories levels recommended per meal. You should take 675 calories per meal if you are making 4 meals per day. "'
    if df['cholesterol_mg'][0]>(0.3/4):
        value_chol = round(df["cholesterol_mg"][0]/0.25,ndigits=2)
        f':warning: Cholesterol: You are having more than the cholesterol levels recommended per meal. You should take 0.3 grams  per day, but careful, your meal already has {value_chol} grams of cholesterol! '
    if df['protein_g'][0]>(50/4):
        value_protein = round(df["protein_g"][0],ndigits=2)
        f':warning:  Protein: You are having more than the protein levels recommended per meal. You should take 63g  per day, but your meal already has {value_protein} grams of protein.'
    if df['carbohydrates_total_g'][0]>(304/4):
        value_carbs = round(df["carbohydrates_total_g"][0],ndigits=2)
        f':warning: Carbohydrates: You are having more than the carbohydrate levels recommended per meal. You should take 410g  per day, but your meal already has {value_carbs} grams of protein.'
    else:
        'The rest of the levels are just ok :muscle:!'

#url = 'https://res.cloudinary.com/sanitarium/image/fetch/q_auto/https://www.sanitarium.com.au/getmedia%2Fae51f174-984f-4a70-ad3d-3f6b517b6da1%2Ffruits-vegetables-healthy-fats.jpg%3Fwidth%3D1180%26height%3D524%26ext%3D.jpg'
#
#st.image(url, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

# Downloading model on cache
model = download_model()
# Sidebar with project info
about = st.sidebar.header(
    'About')
about_text = st.sidebar.write('Food detective is a app build in Streamlit and HerokuApp running a machine learning model')
calorie_ninjas = st.sidebar.header(
    'Calories')
calorie_ninjas_text = st.sidebar.write('We use the [calorie ninjas](https://calorieninjas.com/api) API to check the amount of calories that 100 gr of the food in the photo contain')
calorie_ninjas_logo = st.sidebar.markdown(
    '![](https://i.ibb.co/gg2k4LK/CN.jpg)')

# Main function
st.title('''Hi! Welcome to Food Detective :green_salad: :mag: :eyes:''')
st.subheader('Upload a photo of your meal to know about its nutritional information!:memo:')
st.text('First we need some personal information:')

gender = st.radio('Gender:',('Male', 'Female'))

age = st.slider('Age:', 0, 100, 15)
weight = st.text_input('Weight (kg):', '')


with st.beta_expander("Search image..."):
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        
        imagen = Image.open(uploaded_file)
        st.image(imagen, use_column_width=True)
        new_width  = 64
        new_height = 64
        imagen = imagen.resize((new_width, new_height), Image.ANTIALIAS)
        st.write("")
        imagen = preprocessing_func(imagen)
        #linea para hace predict
        with st.spinner('Please wait! We are inspecting your meal...'):
            category_name_sample = predict_category(model,imagen)
            api_info = get_api_info(category_name_sample)
            api_info_text = get_api_info(category_name_sample)
            df_api = json.loads(api_info_text)
            api_info = convert_data(df_api)
            api_info_transformed = add_statement(api_info)
        st.success(':white_check_mark: Done! You are having...')
        
        f'**{category_name_sample}**!'
        f'Is it actually **{category_name_sample}** ?'
        
        # yesno = st.empty()

        value = st.radio('',('','Yes', 'No'))

        if value == 'No':
            'That is too bad! :cry:'
            'Please let us know what your meal is and we will use this information to keep on learning!'
            real_category = st.text_input('Please enter your meal name here please:', value='')
            if not real_category:
                st.stop()
            
            if gender == 'Female':
                if age > 50:
                    "Here's the nutritional information for your meal"
                    api_info = get_api_info(real_category)
                    api_info_text = get_api_info(real_category)
                    df_api = json.loads(api_info_text)
                    api_info = convert_data(df_api)
                    api_info_transformed = add_statement(api_info)
                    api_info_transformed
                    warnings_women_old(df_api)
                if age < 50:
                    "Here's the nutritional information for your meal"
                    api_info = get_api_info(real_category)
                    api_info_text = get_api_info(real_category)
                    df_api = json.loads(api_info_text)
                    api_info = convert_data(df_api)
                    api_info_transformed = add_statement(api_info)
                    api_info_transformed
                    warnings_women_young(df_api)
                    
            if gender == 'Male':
                if age > 50:
                    "Here's the nutritional information for your meal"
                    api_info = get_api_info(real_category)
                    api_info_text = get_api_info(real_category)
                    df_api = json.loads(api_info_text)
                    api_info = convert_data(df_api)
                    api_info_transformed = add_statement(api_info)
                    api_info_transformed
                    warnings_men_old(df_api)
                if age < 50:
                    "Here's the nutritional information for your meal"
                    api_info = get_api_info(real_category)
                    api_info_text = get_api_info(real_category)
                    df_api = json.loads(api_info_text)
                    api_info = convert_data(df_api)
                    api_info_transformed = add_statement(api_info)
                    api_info_transformed
                    warnings_men_young(df_api)

        if value == 'Yes':
            "That is awesome :sweat_smile: Let's inspect its nutritional information!"
            if gender == 'Female':
                if age > 50:
                    api_info_transformed
                    warnings_women_old(df_api)
                if age < 50:
                    api_info_transformed
                    warnings_women_young(df_api)

            if gender == 'Male':
                if age > 50:
                    api_info_transformed
                    warnings_men_old(df_api)
                if age < 50:
                    api_info_transformed
                    warnings_men_young(df_api)



