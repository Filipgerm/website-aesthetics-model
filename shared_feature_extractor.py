# Shared Feature Extractor for the heads 
from tensorflow import keras
from keras import layers, models, regularizers
from lrn_layer import LRN  

def create_shared_feature_extractor(input_shape=(192, 256, 3), l=0.001):
    im_data = layers.Input(shape=input_shape, dtype='float32', name='im_data')

    conv1 = layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), name='conv1',
                          activation='relu', input_shape=input_shape,
                          kernel_regularizer=regularizers.l2(l))(im_data)

    pool1 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name='pool1')(conv1)
    norm1 = LRN(name='norm1')(pool1)
    # norm1 = layers.BatchNormalization(name='norm1')(pool1)
    drop1 = layers.Dropout(0.1, name='dropout1')(norm1)

    layer1_1 = layers.Lambda(lambda x: x[:, :, :, :48], name='split1_1')(drop1)
    layer1_2 = layers.Lambda(lambda x: x[:, :, :, 48:], name='split1_2')(drop1)

    conv2_1 = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                            activation='relu', padding='same',
                            name='conv2_1', kernel_regularizer=regularizers.l2(l))(layer1_1)

    conv2_2 = layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                            activation='relu', padding='same',
                            name='conv2_2', kernel_regularizer=regularizers.l2(l))(layer1_2)

    conv2 = layers.Concatenate(name='conv_2')([conv2_1, conv2_2])

    pool2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(conv2)
    norm2 = LRN(name="norm2")(pool2)
    # norm2 = layers.BatchNormalization(name="norm2")(pool2)
    drop2 = layers.Dropout(0.1, name='dropout2')(norm2)

    conv3 = layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                          name='conv3', padding='same',
                          kernel_regularizer=regularizers.l2(l))(drop2)
    drop3 = layers.Dropout(0.1, name='dropout3')(conv3)

    layer3_1 = layers.Lambda(lambda x: x[:, :, :, :192], name='split3_1')(drop3)
    layer3_2 = layers.Lambda(lambda x: x[:, :, :, 192:], name='split3_2')(drop3)

    conv4_1 = layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1),
                            activation='relu', padding='same',
                            name='conv4_1', kernel_regularizer=regularizers.l2(l))(layer3_1)

    conv4_2 = layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1),
                            activation='relu', padding='same',
                            name='conv4_2', kernel_regularizer=regularizers.l2(l))(layer3_2)

    conv4 = layers.Concatenate(name='conv_4')([conv4_1, conv4_2])

    layer4_1 = layers.Lambda(lambda x: x[:, :, :, :192], name='split4_1')(conv4)
    layer4_2 = layers.Lambda(lambda x: x[:, :, :, 192:], name='split4_2')(conv4)

    conv5_1 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                            activation='relu', padding='same',
                            name='conv5_1', kernel_regularizer=regularizers.l2(l))(layer4_1)

    conv5_2 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                            activation='relu', padding='same',
                            name='conv5_2', kernel_regularizer=regularizers.l2(l))(layer4_2)

    conv5 = layers.Concatenate(name='conv_5')([conv5_1, conv5_2])

    pool5 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(conv5)

    shared_model = models.Model(inputs=im_data, outputs=pool5, name='shared_feature_extractor')
    return shared_model