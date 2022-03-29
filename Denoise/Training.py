
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend 
import tensorflow as tf
from .ModelDenoise import model

def train(X_train, y_train, X_val,y_val, epochs, batch_size,model):

    model.fit(X_train, y_train, validation_data = (X_val, y_val),epochs = epochs, batch_size = batch_size)




def mix_noise_data():
    pass

# model = unet()
# model.summary()