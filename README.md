# Food Detective :green_salad: :mag: :eyes:


## Table of contents
* [General info](#general-info)
* [Demo](#demo)
* [Site](#site)
* [Technologies](#technologies)
* [Project description](#project-description)


## Project objective

The purpose of this project was to create an app that recognizes the food on your dish and provides its nutritional information along with warnings about the daily intake of each nutrient present in the food.

## Demo :rocket:

You can check it out here :point_right: https://the-food-detective.herokuapp.com/

## Site
### Landing page

![Landing page](./images/landingpage.png)

### Complete your personal information

![Personal information](./images/personalinfo.png)

### Upload an image of your meal

![Meal](./images/imageuploaded.png)

### Get its nutritional information

![Meal](./images/nutritional.png)


## Technologies
Project was created with: 
* Python version:  
* Heroku
* Streamlit


## Project description

### Data source

To carry out the project we used the kaggle's dataset Food 101. It contains 1000 of images of 101 different categories of food, giving a total of 100 000 images. 

![Data](./images/kaggle.png)

The images were presented in different ways:

* All together in in different folders divided by category
* .h5 files of different image sizes

One of the project's objectives was to link the food in the image with its nutritional information.
Therefore we used the API Calorie Ninjas, that contains free nutrition data for 100,000+ foods and beverages.

![Calorie](./images/calorie.png)


### Models trained

We started using different pretrained models such as VGG16, Mobilenet, DenseNet, Resnet50 and EfficientNet. However we weren't very lucky, they presented low accuracy rates between 20% and 39%.

So we decided to try Google Cloud AutoML Vision, a Google Cloud's tool that allows you to derive insights from object detection and image classification, in the cloud or at the edge. 

We uploaded the entire dataset to Google Storage and we trained the model. 
After a few hours we obtained a model that predicted with 91.7% accuracy. Impressive right?
Here we have some of the results obtained. 


Confusion matrix 

![Confusion](./images/confusion.png)

Precision

![Precision](./images/precision.png)


