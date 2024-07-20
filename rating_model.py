# Rating model 
import tensorflow as tf
from keras import layers, initializers, regularizers, models

def create_ratingModel(shared_base, l=0.001):
    flat = layers.Flatten()(shared_base.output)
    fc6 = layers.Dense(1024, activation='relu', name='Ratingfc6',
                       kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                       bias_initializer='zeros',
                       kernel_regularizer=regularizers.l2(l))(flat)
    drop6 = layers.Dropout(0.5, name='RatingDropout6')(fc6)

    fc7 = layers.Dense(512, activation='relu', name='Ratingfc7',
                       kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                       bias_initializer='zeros',
                       kernel_regularizer=regularizers.l2(l))(drop6)
    drop7 = layers.Dropout(0.5, name='RatingDropout7')(fc7)

    fc8 = layers.Dense(1, name='rating_output',
                       kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01),
                       bias_initializer='zeros')(drop7)

    rating_model = models.Model(inputs=shared_base.input, outputs=fc8, name = 'rating_model')

    return rating_model
    # return fc8