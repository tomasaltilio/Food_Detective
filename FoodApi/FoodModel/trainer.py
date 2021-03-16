import numpy as np 
import pandas as pd 
import h5py
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import models, Sequential, layers
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    data_path = '/home/danicanadas/code/Food_Detective_tomasaltilio/Food_Detective/raw_data/food_c101_n10099_r64x64x3.h5'
    test_path = '/home/danicanadas/code/Food_Detective_tomasaltilio/Food_Detective/raw_data/food_test_c101_n1000_r64x64x3.h5'
    data_set = h5py.File(data_path, 'r')
    test_set = h5py.File(test_path, 'r')
    return data_set, test_set

def prepoccess_data(data_set, test_set ):
    X_train, X_val, y_train, y_val = train_test_split(data_set['images'][:], data_set['category'][:], test_size=0.3, random_state=42)
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)        
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    X_test = test_set['images'][:]
    y_test = test_set['category'][:]
    X_test = preprocess_input(X_test)
    y_test = y_test.astype(int)
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=(0, 1.5),
        zoom_range=(0.8, 1.2),
        ) 
    train_flow = datagen.flow(X_train, y_train, batch_size=32)
    return  train_flow, X_val, y_val, X_test, y_test

def build_model():
  efficient = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(64,64,3))
  base_model = set_nontrainable_layers(efficient)
  prediction_layer = layers.Dense(101, activation='softmax')

  model = Sequential([base_model,
                      layers.GlobalAveragePooling2D(),
                      layers.Dropout(0.2),
                      layers.Dense(256, activation='relu'),
                      layers.BatchNormalization(),
                      layers.Dropout(0.1),
                      layers.Dense(128, activation='relu'),
                      layers.BatchNormalization(),
                      layers.Dropout(0.1),
                      prediction_layer])
  return model

def set_nontrainable_layers(model):
  model.trainable = False    
  return model

def compile_model(model):
  model.compile(optimizer = Adam(learning_rate=0.001), 
                  metrics = 'accuracy', 
                  loss='categorical_crossentropy')
  return model
def run_model(model):  
    es = EarlyStopping(patience = 30, restore_best_weights=True)
    model.compile(optimizer = SGD(learning_rate=0.3), 
                    metrics = 'accuracy', 
                    loss='categorical_crossentropy')
    history = model.fit(train_flow, 
            epochs=1000, 
            validation_data=(X_val, y_val),
            callbacks=[es],
            verbose=1, 
            batch_size=32)
    eval = model.evaluate(X_test, y_test)
    return history, eval

data_set, test_set = load_data()
train_flow, X_val, y_val, X_test, y_test = prepoccess_data(data_set, test_set)
model = build_model()
model = compile_model(model)
history, eval = run_model(model)
model.save('model.h5',overwrite=True,include_optimizer=True)