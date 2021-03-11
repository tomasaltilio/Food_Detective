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
from threading import Event
import sys
from progress.bar import Bar


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

def download_model():
    """Function that downloads the model"""
    path = '../ResNET_acc32'
    model_1 = models.load_model(path)
    return model_1

def predict_category(model,image):
    """Function that predicts the category of the image"""
    prediction = model.predict(image)
    category_sample = np.argmax(prediction)
    category_name_sample = categories[category_sample]
    category_name_sample = category_name_sample.decode('UTF-8').capitalize().replace("_", " ")
    category_name_sample
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

    print(response.text)
url = 'https://res.cloudinary.com/sanitarium/image/fetch/q_auto/https://www.sanitarium.com.au/getmedia%2Fae51f174-984f-4a70-ad3d-3f6b517b6da1%2Ffruits-vegetables-healthy-fats.jpg%3Fwidth%3D1180%26height%3D524%26ext%3D.jpg'
st.image(url, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
st.markdown('''# Hi! Welcome to Food Detective :green_salad: :mag: :eyes:''')

'Upload a photo of your meal to know about its nutritional information!'

with st.beta_expander("Search image..."):
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        imagen = Image.open(uploaded_file)
        st.image(imagen, use_column_width=True)
        new_width  = 64
        new_height = 64
        imagen = imagen.resize((new_width, new_height), Image.ANTIALIAS)
        st.write("")
        #linea para hace predict
        'Classifying your meal...'
        imagen = preprocessing_func(imagen)
        my_bar = st.progress(0)
        model = download_model()
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        'Analyzing...'
        st.balloons()
        ':white_check_mark: Ready! You are having...'
        category_name_sample = predict_category(model,imagen)
        api_info = get_api_info(category_name_sample)
        st.write(api_info)
        #api_info