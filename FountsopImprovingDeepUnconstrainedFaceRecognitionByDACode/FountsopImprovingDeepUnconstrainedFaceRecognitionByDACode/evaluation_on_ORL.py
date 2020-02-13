from __future__ import print_function
from tensorflow import set_random_seed
from keras.utils import np_utils
import warnings
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import SGD,Adadelta
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from random import shuffle
import numpy as np
import cv2
import os

from neural_net import *


def load_ORL():
    root="orl_Crop_100_100"
    folders = os.listdir(root)
    nb = len(folders)
    x_data,x_test=[],[]
    y_label, y_test=[], []
    for x in range(nb):
        label=np.zeros(nb)
        label[x]=1
        facesPath=os.listdir(root+"/"+folders[x])
        facesTr=[root+"/"+folders[x]+"/"+f for f in facesPath if (f.endswith(".pgm") or f.endswith(".jpg") or f.endswith(".png"))]
        i=0
        for face in facesTr:
            imgs=cv2.imread(face)
            imgs= cv2.resize(imgs,(100,100))
            img=np.array(imgs)
            i+=1
            if(i not in [2,7,10]):
              x_data.extend([img])
              y_label.extend([label])
            elif(x%2==0 and i==10):
              x_data.extend([img])
              y_label.extend([label])
            else:
              x_test.extend([img])
              y_test.extend([label])
    return (np.asarray(x_data), np.asarray(y_label),np.asarray(x_test), np.asarray(y_test), nb)
  






(x_train, y_train, x_test, y_test,nb_class)=load_ORL()
(x_train, y_train)=mixData(x_train, y_train)
print("train: ",len(x_train), " Test: ",len(x_test))


delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model_aug_orl = model_classifier('model_weights_AUG.h5',nb_class=nb_class,pretraining_nb_class=27)
model_aug_orl.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])


print("training")
hist3=model_s_ORL.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=100, verbose=1)

#Testing
print("evaluation")
score3=model_s_ORL.evaluate(x_test, y_test)
print(score3)

plt.plot(hist3.history['acc'], dashes=[1, 1, 1, 1])
plt.plot(hist3.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show()










model_aug_orl = model_yale27('model_weights_AUG_T.h5',nb_class=nb_class)
model_aug_orl.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])
print("Entrainement du modele")
hist4=model_aug_orl.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=500, verbose=1)

print("evaluation")
score4=model_aug_orl.evaluate(x_test, y_test)
print(score4)

plt.plot(hist4.history['acc'])
plt.plot(hist4.history['val_acc'])

plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show()