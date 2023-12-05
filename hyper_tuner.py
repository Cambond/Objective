import keras_tuner 
from tensorflow import keras 
from keras import backend as K 
from tensorflow.keras import layers, losses 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from keras.layers import *
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

#model tuning
def build(hp): 
    image_size_x=75
    image_size_y=75
    # model hyperparameters setting
    L2_regularizer = hp.Choice('L2', [ 1.5e-5,1.5e-4,1.5e-3,1.5e-2])
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(image_size_x, image_size_y, 1),input_shape=(image_size_x, image_size_y)))
    #self.model.add(RandomFlip()) 
    for i in range( hp.Int("num_conv", min_value=1, max_value=4, step=1)) : 
        # Tune hyperparams of each conv layer separately by using f"...{i}" 
        model.add(BatchNormalization())
        model.add(layers.Conv2D(filters=hp.Int(name=f"filters_{i}", min_value=10, max_value=70, step=10), 
        kernel_size= hp.Int(name=f"kernel_{i}", min_value=3, max_value=7, step=2), 
        padding='same', input_shape=(image_size_x, image_size_y, 1),
        activation=hp.Choice(f"conv_act_{i}",["relu","leaky_relu","sigmoid" ]))) 
    
    model.add(Flatten())    
    for i in range(hp.Int("num_dense", min_value=1, max_value=2, step=1)) : 
        model.add(layers.Dense(hp.Int("neurons", min_value=50, max_value=300, step=40), 
                                   activation=hp.Choice("mlp_activ", ['sigmoid', 'relu']),
                                   kernel_regularizer=l2(L2_regularizer))
                                   ) 
        #model.add(BatchNormalization())
    
    model.add(Dense(5, activation='softmax'))
   
    model.compile(optimizer='adam', 
                loss="categorical_crossentropy", 
                metrics = ['accuracy'])
    learning_rate = hp.Choice('lr', [ 10e-6,10e-5,10e-4,10e-3]) 
    K.set_value(model.optimizer.learning_rate, learning_rate) 
    return model 
 
def Tuning(x_train,y_train,x_val,y_val,epochs=5): 
    tuner = keras_tuner.BayesianOptimization( 
                        hypermodel=build, 
                        objective = "val_accuracy", 
                        max_trials =20, #max candidates to test 
                        )
    tuner.search(x_train, y_train, batch_size=1, epochs=epochs, validation_data=(x_val, y_val))
    #best_model = tuner.get_best_models()[0]
    print(tuner.results_summary(1))




