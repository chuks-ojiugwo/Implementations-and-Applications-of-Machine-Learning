from __future__ import print_function
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
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
from PIL import Image



def load_YALE():
    root="DATA_YALE/yalefacesCrp"
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
        facesTr=[root+"/"+folders[x]+"/"+f for f in facesPath if f.endswith(".pgm")]
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
  





(x_train, y_train, x_test, y_test,nb_class)=load_YALE()
(x_train, y_train)=mixData(x_train, y_train)
print(len(x_train), len(x_test))
model_S_yale = model_classifier('model_weights_SIMP.h5',nb_class=nb_class, pretraining_nb_class=5749)
delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
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

"""##AUGMENTED"""

(x_train, y_train, x_test, y_test,nb_class)=load_YALE()
(x_train, y_train)=mixData(x_train, y_train)
print(len(x_train), len(x_test))
model_aug_yale = model_yale27('model_weights_AUG_T.h5',nb_class=nb_class)
delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
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






(x_train, y_train, x_test, y_test,nb_class)=load_ORL()
(x_train, y_train)=mixData(x_train, y_train)
print("train: ",len(x_train), " Test: ",len(x_test))
model_s_ORL = model_yale('model_weights_SIMP.h5',nb_class=nb_class)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0)
model_s_ORL.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])

"""###Eval"""

print("Entrainement du modele")
hist3=model_s_ORL.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=100, verbose=1)

score3=model_s_ORL.evaluate(x_test, y_test)
print(score3)

plt.plot(hist3.history['acc'], dashes=[1, 1, 1, 1])
plt.plot(hist3.history['val_acc'])
#plt.title('cross validation training on ORL  / 500 iteration')
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""##AUGMENTE"""

set_random_seed(1234)
(x_train, y_train, x_test, y_test,nb_class)=load_ORL()
(x_train, y_train)=mixData(x_train, y_train)
print("train: ",len(x_train), " Test: ",len(x_test))
model_aug_orl = model_yale27('model_weights_AUG_T.h5',nb_class=nb_class)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model_aug_orl.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])

print("Entrainement du modele")
hist4=model_aug_orl.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=32, epochs=500, verbose=1)

"""###eval"""

score4=model_aug_orl.evaluate(x_test, y_test)
print(score4)

plt.plot(hist4.history['acc'])
plt.plot(hist4.history['val_acc'])
#plt.title('courbe de la perte DataSet avec augmentation / 1000 itération')
plt.ylabel('accuracy')
plt.xlabel('iteration')
plt.legend(['train', 'test'], loc='upper left')
plt.show()