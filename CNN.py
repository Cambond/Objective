from tensorflow import keras
from keras.layers import *
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pickle as pkl
import os
import numpy as np
import math
import matplotlib.pyplot as plt 
import traceback
from datetime import datetime
import statistics
from sklearn import metrics

def lr_step_decay(epoch,current_lr):
    drop = 0.5
    epochs_drop = 50
    initial_lr = 2e-2
    step_lr = initial_lr * math.pow(drop, math.floor(1+epoch)/epochs_drop)
    return step_lr

class CNN:
    # giving the work path with four pickled data files, then it will load the data set
    def __init__(self,path):
        self.path = path
        try:
            self.__load_train(path)
        except:
            pass
        try:
            self.__load_test(path)
        except:
            pass
    
    def __load_train(self,path):
        subdir_folders = []
        train_in = []
        train_out = []
        for sub_dir in os.listdir(path):
            if 'train_in' in sub_dir:
                train_in.append(sub_dir)
            if 'train_out' in sub_dir:
                train_out.append(sub_dir)
        if len(train_in)*len(train_out) != 1:
            print('Warning:please check the number of dataset')  
        train_in = train_in[0]
        train_out = train_out[0]
        try:
            if not (train_in.replace('train_in','')==train_out.replace('train_out','')):
                print("data set mismatch.") 
        except: 
            print("data loading fails") 

        self.Train_in = np.array(pkl.load(open(path + '/' + train_in, "rb")))
        self.Train_out = np.array(pkl.load(open(path + '/' + train_out, "rb")))

        print("load traning input set with size ",self.Train_in.shape)
        print("load traning output set with size ",self.Train_out.shape)
        print("load testing input set with size ",self.Test_in.shape)
        print("load testing output set with size ",self.Test_out.shape)  
        
    def __load_test(self,path):
        subdir_folders = []
        test_in = []
        test_out = []
        for sub_dir in os.listdir(path):
            if 'test_in' in sub_dir:
                test_in.append(sub_dir)
            if 'test_out' in sub_dir:
                test_out.append(sub_dir)  
        if len(test_in)*len(test_out) != 1:
            print('Warning:please check the number of dataset')  
        test_in = test_in[0]
        test_out = test_out[0]
        try:
            if not (test_in.replace('test_in','')==test_out.replace('test_out','')):
                print("data set mismatch.") 
        except: 
            print("data loading fails") 

        self.Test_in = np.array(pkl.load(open(path + '/' + test_in, "rb")))
        self.Test_out = np.array(pkl.load(open(path + '/' + test_out, "rb")))

        print("load testing input set with size ",self.Test_in.shape)
        print("load testing output set with size ",self.Test_out.shape)

    # defining the architecture of CNN model 
    def model_v1(self):
        self.model_name = 'CNN_original'

        # model hyperparameters setting
        conv_size_1 = 8
        conv_size_2 = 16
        #conv_size_3 = 64
        conv_size_3 = 32
        dense_size = 256
        L2_regularizer = 1.5e-3

        # sequential structure
        image_size_x = self.Train_in.shape[1] 
        image_size_y = self.Train_in.shape[2] 
        output_size = self.Train_out.shape[1]
        self.model = keras.models.Sequential()
        self.model.add(Reshape(target_shape=(image_size_x, image_size_y, 1),input_shape=(image_size_x, image_size_y)))
        #self.model.add(RandomFlip()) 
        self.model.add(Conv2D(conv_size_1,
                         kernel_size=3,
                         padding='same',
                         input_shape=(image_size_x, image_size_y, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(conv_size_2,
                         kernel_size=3,
                         padding='same',
                         input_shape=(image_size_x, image_size_y, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(conv_size_3,
                         kernel_size=3,
                         padding='same',
                         input_shape=(image_size_x, image_size_y, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(dense_size, kernel_regularizer=l2(L2_regularizer), activation='relu'))
        #self.model.add(Dense(dense_size, kernel_regularizer=l2(L2_regularizer), activation='relu'))
        #self.model.add(Dense(dense_size, kernel_regularizer=l2(L2_regularizer), activation='relu'))
        self.model.add(Dense(output_size, activation='softmax'))
        
    def model_v2(self):
        self.model_name = 'CNN_original_tuned'

                # model hyperparameters setting
        conv_size_1 = 40
        conv_size_2 = 60
        conv_size_3 = 40
        dense_size = 170
        L2_regularizer = 1.5e-5

        # sequential structure
        image_size_x = self.Train_in.shape[1] 
        image_size_y = self.Train_in.shape[2] 
        output_size = self.Train_out.shape[1]
        self.model = keras.models.Sequential()
        self.model.add(Reshape(target_shape=(image_size_x, image_size_y, 1),input_shape=(image_size_x, image_size_y)))
        #self.model.add(RandomFlip()) 
        self.model.add(Conv2D(conv_size_1,
                         kernel_size=3,
                         padding='same',
                         input_shape=(image_size_x, image_size_y, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(conv_size_2,
                         kernel_size=7,
                         padding='same',
                         input_shape=(image_size_x, image_size_y, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(conv_size_3,
                         kernel_size=5,
                         padding='same',
                         input_shape=(image_size_x, image_size_y, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(dense_size, kernel_regularizer=l2(L2_regularizer), activation='relu'))
        self.model.add(Dense(output_size, activation='softmax'))

    # carry out the model training
    def train_v1(self,epochs = 1,batch_size = 1):
        # training parameters
        validationset_ratio = 0.1
        steps_per_epoch = None
        optimizer = keras.optimizers.Adam(learning_rate=10e-4, epsilon=2e-2) 
        checkpoint = ModelCheckpoint(self.path + "CNN_weights_%s.hdf5" % datetime.today().strftime("%b_%d_%H%M"),
                                    monitor="val_accuracy", verbose=1,
                                    save_best_only=True, mode="auto", save_freq="epoch",)
        learning_rate = LearningRateScheduler(lr_step_decay)
        stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100)
        callbacks_list = [checkpoint, learning_rate, stopping]

        # training codes
        loss_fn = keras.losses.CategoricalCrossentropy()
        self.model.compile(loss=loss_fn,
                           optimizer = optimizer,
                           metrics='accuracy')

        print("-------------------model training-----------------------")
        self.model.fit(self.Train_in,
          self.Train_out,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=validationset_ratio,
          steps_per_epoch=steps_per_epoch,
          callbacks=[callbacks_list],
          )
        print("--------------------model info--------------------")
        print("Model name: %s" %self.model_name)
        print(self.model.summary())
        
    def save_model(self,savepath = None):
        if savepath == None:
            savepath = self.path
        try:
            self.model.save(savepath + '/%s_%s.h5' % (self.model_name,datetime.today().strftime("%b_%d_%H%M")))
            print("model [%s] is saved at %s" %(self.model_name,self.path))
        except:
            print("warning: model is not saved")
            traceback.print_exc()
            
 
        
    def save_model(self,savepath = None):
        if savepath == None:
            savepath = self.path
        try:
            self.model.save(savepath + '/%s_%s.h5' % (self.model_name,datetime.today().strftime("%b_%d_%H%M")))
            print("model [%s] is saved at %s" %(self.model_name,self.path))
        except:
            print("warning: model is not saved")
            traceback.print_exc()

    def load_model(self,name = None):
        try:
            if name == None:
                 temp = os.listdir(self.path)
                 for name in temp:
                    if '.h5' in name:
                        break       
                 if '.h5' in name:
                     self.model = load_model(self.path + '/' + name)
                     print("Model %s is loaded sucessfully" % name)
                     self.model.summary()
                 else:
                     print("warning: Model is not founded" % name)

        except:
            print("warning: model is not loaded")
            traceback.print_exc()
            
    def model_predict(self,data=None):
        if data == None:
            data = self.Test_in
        return self.model.predict(data)
    
    def benchmark(self):
        print('---------------model evaluation------------------')
        Pred = self.model_predict()
        Pred = Pred.tolist()
        M = np.zeros((5,5))
        for i in np.arange(len(self.Test_in)):
            temp1 = Pred[i]
            temp2 = self.Test_out[i]
            temp2 = temp2.tolist()
            temp1 = temp1.index(max(temp1))
            temp2 = temp2.index(max(temp2))
            M[temp1,temp2] = M[temp1,temp2] + 1 
        plt.subplot(121)
        plt.imshow(M)
        plt.xlabel('Real label') 
        plt.xticks([0,1,2,3,4],['--','-','0','+','++'])
        plt.ylabel('Prediction') 
        plt.yticks([0,1,2,3,4],['--','-','0','+','++'])
        fidelity = M.trace()/np.sum(M)
        plt.title('Fidelity = %.2f' % fidelity)
        plt.colorbar()
        print('Fidelity: %.2f' % fidelity)
        score = self.model.evaluate(self.Test_in,self.Test_out)
        print('Test loss:', score[0]) 
        print('Test accuracy:', score[1])
        Fpr = []
        Tpr = []
        AUC = []
        self.Test_out = np.array(self.Test_out)
        Pred = np.array(Pred)
        for ind in np.arange(len(Pred[0])):
            fpr,tpr,_ = metrics.roc_curve(self.Test_out[:,ind],Pred[:,ind])
            try:
                AUC.append(metrics.roc_auc_score(self.Test_out[:,ind],Pred[:,ind]))
            except:
                AUC.append(0)
            Fpr.append(fpr)
            Tpr.append(tpr)
        plt.subplot(122)    
        for ind in np.arange(len(Fpr)):
            plt.plot(np.array(Fpr[ind]),np.array(Tpr[ind]),label=str(ind)+" AUC=%.2f" % AUC[ind])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.show()
        return M,fidelity,score[0],score[1],AUC

    def predict(self,input_data):
        result = self.model.predict(input_data)
        return result

    def data_loader(self,file_name):
        result = np.array(pkl.load(open(path + '/' + name, "rb")))
        return result






