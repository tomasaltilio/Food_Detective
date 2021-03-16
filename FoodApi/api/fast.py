
import pandas as pd
import h5py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import models
import numpy as np 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from fastapi.encoders import jsonable_encoder
from PIL import Image
from typing import List


from fastapi.responses import HTMLResponse

from io import BytesIO
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
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
# http://127.0.0.1:8000/predict_fare/?key=2012-10-06 12:10:20.0000001&pickup_datetime=2012-10-06 12:10:20 UTC&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    
    return image

def numpy_fun(image: Image.Image):
    image = np.asarray(image.resize((64, 64)))[..., :3]
    preproc_img = np.expand_dims(image, axis = 0)
    preproc_img = preprocess_input(preproc_img)
    return preproc_img

@app.post("/predict/image/")
async def predict_api(files: UploadFile = File(...)):
    extension = files.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await files.read())
    preproc_img = numpy_fun(image )
    #load model
    model = models.load_model('model.h5')
    #make pred
    prediction = model.predict(preproc_img)
    category_sample = np.argmax(prediction)
    category_name_sample = categories[category_sample]
    category_name_sample = category_name_sample.decode('UTF-8').capitalize().replace("_", " ")
    #return {"filenames": files.filename }
    return {'prediction': category_name_sample}



@app.get("/")
async def main():
    content = """
<body>
<form action="/predict/image/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</body>
    """
    return HTMLResponse(content=content)


