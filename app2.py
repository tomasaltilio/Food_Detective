import streamlit as st
from PIL import Image
import requests
from Food_Detective.params import categories, background_image, radio_button, responsive
from Food_Detective.utils import preprocessing_func, download_model, predict_category,\
    get_api_info, convert_data, add_statement, warnings_kids_2to3, warnings_women_4to8,\
    warnings_women_9to13, warnings_women_14to18, warnings_women_19to30, warnings_women_31to50,\
    warnings_women_morethan51, warnings_men_4to8, warnings_men_9to13, warnings_men_14to18,\
    warnings_men_19to30, warnings_men_31to50, warnings_men_morethan51

import json


# Main function
def button(category_name_sample):
    value = st.radio('',('','Yes', 'No'))

    if value == 'No':
        st.write('''That is too bad! :cry:''')
        st.write('''Please let us know what your meal is and we will use this information to keep on learning!''')
        real_category = st.text_input('Your food:', value='')
        if not real_category:
            st.stop()
                
        api_info = get_api_info(real_category)
        if api_info == "Error":
            st.write('''Sorry! :cry: we do not have the requested food in our database ''')
            st.stop()
        else:
            api_info_text = get_api_info(real_category)
            df_api = json.loads(api_info_text)
            api_info = convert_data(df_api)
            api_info_transformed = add_statement(api_info)

        if gender == 'Female':
                if 2 <= age <= 3:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_kids_2to3(df_api)

                if 4 <= age <= 8:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_women_4to8(df_api)

                if 9 <= age <= 13:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_women_9to13(df_api)

                if 14 <= age <= 18:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_women_14to18(df_api)

                if 19 <= age <= 30:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_women_19to30(df_api)

                if 31 <= age <= 50:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_women_31to50(df_api)

                if age >= 51:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_women_morethan51(df_api)

                    
        if gender == 'Male':
                if 2 <= age <= 3:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_kids_2to3(df_api)

                if 4 <= age <= 8:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_men_4to8(df_api)

                if 9 <= age <= 13:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_men_9to13(df_api)

                if 14 <= age <= 18:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_men_14to18(df_api)

                if 19 <= age <= 30:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_men_19to30(df_api)

                if 31 <= age <= 50:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_men_31to50(df_api)

                if age >= 51:
                    f"Here's the nutritional information for your meal for 100 grams of **{real_category}**"
                    api_info_transformed
                    warnings_men_morethan51(df_api)

    if value == 'Yes':
        api_info = get_api_info(category_name_sample)
        if api_info == "Error":
            st.write('''Sorry! :cry: we do not have the requested food in our database ''')
            st.stop()
        api_info_text = get_api_info(category_name_sample)
        df_api = json.loads(api_info_text)
        api_info = convert_data(df_api)
        api_info_transformed = add_statement(api_info)
        st.write('''That is awesome :sweat_smile:''')
        st.write('''Let's inspect it's nutritional information!''')
        if gender == 'Female':
                if 2 <= age <= 3:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_kids_2to3(df_api)

                if 4 <= age <= 8:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_women_4to8(df_api)

                if 9 <= age <= 13:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_women_9to13(df_api)

                if 14 <= age <= 18:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_women_14to18(df_api)

                if 19 <= age <= 30:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_women_19to30(df_api)

                if 31 <= age <= 50:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_women_31to50(df_api)

                if age >= 51:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_women_morethan51(df_api)

        if gender == 'Male':
                if 2 <= age <= 3:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_kids_2to3(df_api)

                if 4 <= age <= 8:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_men_4to8(df_api)

                if 9 <= age <= 13:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_men_9to13(df_api)

                if 14 <= age <= 18:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_men_14to18(df_api)

                if 19 <= age <= 30:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_men_19to30(df_api)

                if 31 <= age <= 50:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_men_31to50(df_api)

                if age >= 51:
                    f"Here's the nutritional information for your meal for 100 grams of **{category_name_sample}**"
                    api_info_transformed
                    warnings_men_morethan51(df_api)


# Site configuration
st.set_page_config(
    page_title="The Food Detective",  
    page_icon="🥗",
    layout="centered", 
    initial_sidebar_state="collapsed") 

# Background image
st.write(f'<style>{background_image}</style>', unsafe_allow_html=True)
st.write(f'<style>{responsive}</style>', unsafe_allow_html=True)

# Setting up radio buttons empty by default
st.write(f'<style>{radio_button}</style>', unsafe_allow_html=True)

# Downloading model on cache
interpreter = download_model()

# Main section
st.title('''Hi! Welcome to Food Detective :green_salad: :mag: :eyes:''')
st.subheader(
    'Upload a photo of your meal to know about its nutritional information!:memo:')
'First we need some personal information:'

# User inputs
gender = st.radio('Gender:', ('', 'Male', 'Female'))
age = st.slider('Age:', 0, 100, 15)

weight = st.text_input('Weight (kg):', '')

# Sidebar with project info
about = st.sidebar.header(
    'About')
about_text = st.sidebar.write(
    'Food detective is an app built using Streamlit and deployed in Heroku, running a Deep Learning model to recognize and classify food images.')
food_data = st.sidebar.write(
    f'The app is able to classify images into 101 different [food categories](https://www.kaggle.com/kmader/food41).')
calorie_ninjas = st.sidebar.header(
    'Nutrition')
calorie_ninjas_text = st.sidebar.write(
    'We use the [calorie ninjas](https://calorieninjas.com/api) API to check nutritional information and give you some health tips')
physician_community_for_responsible_medicine = st.sidebar.write(
    'We use the research provided by the [medical community for responsible medicine]\
        (https://www.dietaryguidelines.gov/sites/default/files/2020-12/Dietary_Guidelines_for_Americans_2020-2025.pdf)\
            to calculate the maximum [calorie intake per meal](https://www.pcrm.org/good-nutrition/nutrition-programs-policies/2020-2025-dietary-guidelines).')



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
        st.success(':white_check_mark: Done! You are having...')
        if api_info == "Error":
            f'**{category_name_sample}**!'
            f'Is it actually **{category_name_sample}** ?'
            button(category_name_sample)
        else :
            f'**{category_name_sample}**!'
            f'Is it actually **{category_name_sample}** ?'
            button(category_name_sample)


