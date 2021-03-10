import requests
import h5py
import numpy as np
from tensorflow.keras import models
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input

def preprocessing_func(image):
    '''Function that preprocesses the input''''
    preproc_img = img_to_array(image)
    preproc_img = np.expand_dims(image, axis = 0)
    preproc_img = preprocess_input(image)
    return preproc_img
        
def download_model():
    """Function that downloads the model"""
    path = 'PONER PATH DEL MODELO AQUI'
    model = models.load_model(path)
    return model

def predict_category(model):
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
    query = f'100g {category_name_sample}'
    response = requests.get(api_url + query, headers={'X-Api-Key': 'Xy0klDLLr8umMOU3GBTv8g==ixVHYHMXsQA3CJT7'})
    if response.status_code == requests.codes.ok:
        print(response.text)
    else:
        print("Error:", response.status_code, response.text)

    print()

#image = load_img(f'/content/gdrive/My Drive/Fooddetective/{huevos_rancheros.jpg}', color_mode='rgb', target_size=(64,64,3))


if __name__ == '__main__':
    model = download_model()
    category_name_sample = predict_category(model)
    api_info = get_api_info(category_name_sample)
