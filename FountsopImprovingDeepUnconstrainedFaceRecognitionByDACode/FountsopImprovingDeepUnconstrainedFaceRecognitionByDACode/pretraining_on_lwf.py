from __future__ import print_function
from tensorflow import set_random_seed
from keras.utils import np_utils
import warnings
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import SGD,Adadelta
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from random import shuffle
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import cv2
import os

from neural_net import *






def load_trainSet(rootDir, dataset="augmented"):
    root=rootDir
    folders = os.listdir(root)
    nb = len(folders)
    x_data=[]
    y_label=[]
    for x in range(nb):
        label=np.zeros(nb)
        label[x]=1
        facesPath=os.listdir(root+"/"+folders[x])
        faces=[root+"/"+folders[x]+"/"+f for f in facesPath if (f.endswith(".png") or f.endswith(".jpg"))]
        for face in faces:
            imgs=cv2.imread(face)
            img=np.zeros([100,100,3])
            if(os.path.getsize(face)!=0):
              img[:,:,0] = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
              img[:,:,1] = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
              img[:,:,2] = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
              x_data.extend([img])
              y_label.extend([label])
    return (np.asarray(x_data), np.asarray(y_label), nb)


#This routine shuffle the dataset
def mixData(xs,ys):
    xys=[]
    for i in range(0,len(xs)-1):
        xys.append((xs[i],ys[i]))
    shuffle(xys)
    x2,y2=[],[]
    for (x,y) in xys:
        x2.append(x)
        y2.append(y)
    return (np.asarray(x2), np.asarray(y2))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-i', '--Dataset', default='CROPPED_LFW_AUGMENTED', type=str, help='')
    parser.add_argument('-o', '--Train_mode', default='LFW_AUGMENTED', type=str, help='')

    args=parser.parse_args()
    dataset = args.Dataset
    training_mode =  args.Train_mode

    if(training_mode=="LFW_AUGMENTED"):
        """Training with Augmented Dataset"""
        (x_train, y_train, nb_class)=load_trainSet(dataset)
        (x_train, y_train)=mixData(x_train[:12601], y_train[:12601])
        model_aug = conv_model(input_shape=x_train[0].shape , nb_person=len(y_train[0]))
        delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        model_aug.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])

        print("Training on augmented dataset")
        hist=model_aug.fit(x_train, y_train, validation_split=0.10, batch_size=64, epochs=1000, verbose=1)
        model_simp.save_weights('model_weights_AUG.h5')



    if(training_mode=="LFW_SIMPLE"):
        """Training with LWF without agumentation"""
        (x_train, y_train, nb_class)=load_trainSet(dataset,"simple")
        (x_train, y_train)=mixData(x_train[:12601], y_train[:12601])
        model_simp = conv_model(input_shape=x_train[0].shape , nb_person=len(y_train[0]))
        delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        model_simp.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])

        print("Training on simple dataset")
        hist=model_simp.fit(x_train, y_train, validation_split=0.1, batch_size=50, epochs=1000, verbose=1)
        model_simp.save_weights('model_weights_SIMPLE.h5')

