import streamlit as st
from PIL import Image
import requests
from Food_Detective.params import categories, background_image, radio_button
from Food_Detective.utils import preprocessing_func, download_model, predict_category,\
    get_api_info, convert_data, add_statement, warnings_men_young, warnings_men_old,\
    warnings_women_old, warnings_women_young
import json


# Site configuration
st.set_page_config(
    page_title="The Food Detective",  
    page_icon="🥗",
    layout="centered", 
    initial_sidebar_state="collapsed") 

# Background image
st.write(f'<style>{background_image}</style>', unsafe_allow_html=True)

# Setting up radio buttons empty by default
st.write(f'<style>{radio_button}</style>', unsafe_allow_html=True)

# Downloading model on cache
interpreter = download_model()

# Main section
st.title('''Hi! Welcome to Food Detective :green_salad: :mag: :eyes:''')
st.subheader(
    'Upload a photo of your meal to know about its nutritional information!:memo:')
st.text('First we need some personal information:')

# User inputs
gender = st.radio('Gender:', ('', 'Male', 'Female'))
age = st.slider('Age:', 0, 100, 15)

weight = st.text_input('Weight (kg):', '')

# Sidebar with project info
about = st.sidebar.header(
    'About')
about_text = st.sidebar.write(
    'Food detective is a app build in Streamlit and HerokuApp running a machine learning model')
calorie_ninjas = st.sidebar.header(
    'Calories')
calorie_ninjas_text = st.sidebar.write(
    'We use the [calorie ninjas](https://calorieninjas.com/api) API to check the amount of calories that 100 gr of the food in the photo contain')
calorie_ninjas_logo = st.sidebar.markdown(
    '![](https://i.ibb.co/gg2k4LK/CN.jpg)')


# User uploads image
with st.beta_expander("Search image..."):
    uploaded_file = st.file_uploader(
        "Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Preprocessing image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        st.write("")
        input_data = preprocessing_func(image, interpreter)
        # Prediction with loader
        with st.spinner('Please wait! We are inspecting your meal...'):
            category_name_sample = predict_category(interpreter, input_data)
            api_info = get_api_info(category_name_sample)
            api_info_text = get_api_info(category_name_sample)
            df_api = json.loads(api_info_text)
            api_info = convert_data(df_api)
            api_info_transformed = add_statement(api_info)
        st.success(':white_check_mark: Done! You are having...')

        f'**{category_name_sample}**!'
        f'Is it actually **{category_name_sample}** ?'

        # Validates with user if prediction is right and outputs nutritional table along with health warnings
        value = st.radio('', ('', 'Yes', 'No'))

        if value == 'No':
            st.write('''That is too bad! :cry:''')
            st.write(
                '''Please let us know what your meal is and we will use this information to keep on learning!''')
            real_category = st.text_input('Your food:', value='')
            if not real_category:
                st.stop()

            api_info = get_api_info(real_category)
            api_info_text = get_api_info(real_category)
            df_api = json.loads(api_info_text)
            api_info = convert_data(df_api)
            api_info_transformed = add_statement(api_info)

            if gender == 'Female':
                if age > 50:
                    "Here's the nutritional information for your meal"
                    api_info_transformed
                    warnings_women_old(df_api)

                if age < 50:
                    "Here's the nutritional information for your meal"
                    api_info_transformed
                    warnings_women_young(df_api)

            if gender == 'Male':
                if age > 50:
                    "Here's the nutritional information for your meal"
                    api_info_transformed
                    warnings_men_old(df_api)
                if age < 50:
                    "Here's the nutritional information for your meal"
                    api_info_transformed
                    warnings_men_young(df_api)

        if value == 'Yes':
            st.write('''That is awesome :sweat_smile:''')
            st.write('''Let's inspect it's nutritional information!''')
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
