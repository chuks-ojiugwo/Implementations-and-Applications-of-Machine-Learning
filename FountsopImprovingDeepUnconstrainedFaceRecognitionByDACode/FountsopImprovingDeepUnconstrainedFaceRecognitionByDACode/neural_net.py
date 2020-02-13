#******************* MODEL***********************************************
def conv_model(input_shape=(100,100,3), nb_person=10):
    img_input = Input(shape=input_shape)
    #Block 1 avec 64 filtre
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1')(img_input)
    x=BatchNormalization()(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x= Dropout(0.25)(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2')(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x= Dropout(0.25)(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2')(x)
    x=BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x= Dropout(0.25)(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3')(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    x= Dropout(0.25)(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3')(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    x= Dropout(0.25)(x)

    #Classifieur du modele
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x= Dropout(0.25)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(nb_person, activation='softmax', name='fc8')(x)
    model = Model(img_input, x)
    return model




def model_classifier(w_file, nb_class=10,retraining_nb_class=5749):
  model = conv_model(nb_person=retraining_nb_class)
  model.load_weights(w_file)
  img_input =model.input
  for layer in model.layers[0:-3]:
    layer.trainable = False
  x = model.get_layer("pool5").output
  x = Flatten(name='flatten')(x)
  x = Dense(4096, activation='relu', name='fc6')(x)
  x= Dropout(0.25)(x)
  x = Dense(4096, activation='relu', name='fc7')(x)
  x = Dense(nb_class, activation='softmax', name='fc8')(x)
  model2 = Model(img_input, x)
  return model2