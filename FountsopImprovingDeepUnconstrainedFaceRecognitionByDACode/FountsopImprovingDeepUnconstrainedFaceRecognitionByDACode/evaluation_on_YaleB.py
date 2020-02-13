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

def load_YALE():
    root="yalefacesCrp"
    folders = os.listdir(root)
    nb = len(folders)
    x_data,x_t=[],[]
    y_label, y_t=[], []
    for x in range(nb):
        label=np.zeros(nb)
        label[x]=1
        facesPath=os.listdir(root+"/"+folders[x])
        facesTr=[root+"/"+folders[x]+"/"+f for f in facesPath if f.endswith(".png")]
        for face in facesTr:
            imgs=cv2.imread(face)
            imgs= cv2.resize(imgs,(100,100))
            img=np.array(imgs)
            if(not face.endswith("t.png")):
              x_data.extend([img])
              y_label.extend([label])
            else:
              x_t.extend([img])
              y_t.extend([label])   
    return (np.asarray(x_data), np.asarray(y_label),np.asarray(x_t), np.asarray(y_t), nb)



(x_train, y_train, x_test, y_test,nb_class)=load_YALE()
(x_train, y_train)=mixData(x_train, y_train)
print(len(x_train), len(x_test))
delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

model_S_yale = model_classifier('model_weights_SIMP.h5',nb_class=nb_class, pretraining_nb_class=5749)
model_S_yale.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])

print("Model training")
hist=model_S_yale.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=500, verbose=1)

print("Testing")
score1=model_S_yale.evaluate(x_test, y_test)
print(score1)

plt.plot(hist.history['acc'],dashes=[1, 1, 1, 1])
plt.plot(hist.history['val_acc'])
plt.title('accuracy on YaleB without Data augmentation')
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show()







model_aug_yale = model_classifier('model_weights_AUG_T.h5',nb_class=nb_class,pretraining_nb_class=27)
model_aug_yale.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])

print("Entrainement du modele")
hist2=model_aug_yale.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=500, verbose=1)
score2=model_aug_yale.evaluate(x_test, y_test)
print(score2)

plt.plot(hist2.history['acc'] ,dashes=[1, 1, 1, 1])
plt.plot(hist2.history['val_acc'])
plt.title('accuracy on YaleB without Data augmentation')
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

