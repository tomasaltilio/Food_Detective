# Food Detective :green_salad: :mag: :eyes:
 
 Food Detective is an app that recognizes the food on your dish by uploading an image and provides the user its nutritional information along with warnings about the daily intake of each nutrient present in the food.


## Table of contents
* [General info](#general-info)
* [Problem to solve](#problem-to-solve)
* [Demo](#demo)
* [Technologies](#technologies)
* [Project description](#project-description)
* [Site](#site)
* [Bug Or Feature Request](#bug-or-feature-request)


## General Info

This project was made to present as a Final Project for LeWagon's DataScience Bootcamp (Batch 561). 

Its purpose remains on bringing together all the components learnt during the bootcamp on a real open-ended problem.

## Problem to solve
🤔 Did you ever wanted to know how healthy is the food you are about to eat? 

:sleepy:    Did you quickly resign on the idea after realizing it takes a lot of time?

This is what we asked ourselves, and we came up with this idea! :bulb:

How about you upload an image of what you are about to eat and get its nutritional information immediatly? 
Great idea right? 
Let's checkout how we solved it! :rocket:



## Demo

You can visit the web app here :point_right:    https://the-food-detective.herokuapp.com/


## Technologies
Project was created with: 
* Python version:  
* Kaggle Dataset Food 101
* Calorie Ninjas API
* Heroku
* Streamlit


## Project description

### Data source

To carry out the project we used the Kaggle's dataset Food 101. It contains 1000 images of 101 different food categories, giving a total of 100 000 images. 

![Data](./images/kaggle.png)

The images were presented in different ways:

* All together in different folders according to their food category 
* .h5 files of different size images

One of the project's objectives was to link the food in the image with its nutritional information.

Therefore we used Calorie Ninjas, an API that contains free nutrition data for 100,000+ foods and beverages.

![Calorie](./images/calorie.png)


### Models trained

We started using different pretrained models such as **VGG16, Mobilenet, DenseNet, Resnet50** and **EfficientNet**.
Eventhough we modified them in order to get higher accuracies, we weren't so lucky. We obtained low accuracy rates, between 20% and 39%.

We decided then to try **Google Cloud AutoML Vision**, a Google Cloud's tool that allows you to derive insights from object detection and image classification, in the cloud or at the edge. 

We uploaded the entire dataset to Google Cloud Storage, listed the images and their labels, and trained an AutoML Vision Edge model. 

After a few hours we obtained a model that predicted with **91.7%** accuracy. Impressive right? 🤯

Here we have some of the results obtained. 


### Confusion matrix 

![Confusion](./images/confusion.png)

### Precision

![Precision](./images/precision.png)


### Building the app

In order to let anyone use this application, we decided to deploy it on Heroku, a cloud Platform As a Service that allows us to develop, run, and manage the application without the complexity of building and maintaining the infrastructure typically associated with developing and launching an app.


### Visualisation 

To visualize the app we used Streamlit, a framework that allows us to turn the data scripts into a shareable web app. 


## Site
### Landing page
Complete your personal information :pencil:

![Landing page](./images/landingpage.png)

### Upload an image of your meal 📸

![Meal](./images/uploadedphoto.png)

### Get its nutritional information
You will get warnings if you are exceeding recommended levels of different nutritional components


## Bug Or Feature Request

If you find a bug (the website couldn't handle the query and/or gave undesired results), kindly open an issue [here](https://github.com/tomasaltilio/Food_Detective/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/tomasaltilio/Food_Detective/issues/new). Please include sample queries and their corresponding results.



## Team

[![Daniel Cañadas](https://avatars.githubusercontent.com/u/77625064?s=400&u=d06d5bb2d299cfbaf072fae4d9c7abbf1b906ce2&v=4)](https://github.com/dkj1711)  | [![Julieta Hocsman](https://avatars.githubusercontent.com/u/72712108?s=460&u=4fa30b93353b0ca96b50d9cf70b81be90bb8d89e&v=4)](https://github.com/julietahocsman/)  | [![Santiago Caride](https://avatars.githubusercontent.com/u/17711221?s=400&v=4)](https://github.com/carmaiquel/)  | [![Tomas Altilio](https://avatars.githubusercontent.com/u/65192458?s=460&u=0b40d587d88faa7cbeb4353b2c09e61b1c80419b&v=4)](https://github.com/tomasaltilio/)
---|---|---|---
[Daniel Cañadas ](https://github.com/dani) |[Julieta Hocsman](https://github.com/julietahocsman/) |[Santiago Caride](https://github.com/carmaiquel/) |[Tomas Altilio](https://github.com/tomasaltilio/)
