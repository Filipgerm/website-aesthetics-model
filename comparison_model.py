# Comparison model 
from keras import layers, regularizers, models, backend, initializers
import tensorflow as tf
import numpy as np
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

def create_comparisonModel(shared_base, l=0.001):
    input_shape = shared_base.input_shape[1:]
    
    input_a = layers.Input(shape=input_shape, dtype='float32', name='input_a')
    input_b = layers.Input(shape=input_shape, dtype='float32', name='input_b')

    output_a = shared_base(input_a)
    output_b = shared_base(input_b)

    flat_a = layers.Flatten()(output_a)
    flat_b = layers.Flatten()(output_b)
        # flat = layers.Flatten()(shared_base.output)

    fc6_a = layers.Dense(1024, activation='relu', name='Comparisonfc6_a', kernel_regularizer=regularizers.l2(l),
                          kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(flat_a)
    drop6_a = layers.Dropout(0.5, name='ComparisonDropout6_a')(fc6_a)
    fc7_a = layers.Dense(128, activation='relu', name='Comparisonfc7_a', kernel_regularizer=regularizers.l2(l),
                         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(drop6_a)
    drop7_a = layers.Dropout(0.5, name='ComparisonDropout7_a')(fc7_a)
    # fc8_a = layers.Dense(128, activation='relu', name='Comparisonfc8_a',  kernel_regularizer=regularizers.l2(l),
    #                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(drop7_a)
    # drop8_a = layers.Dropout(0.1, name='ComparisonDropout8_a')(fc8_a)
    fc9_a = layers.Dense(1, name='Comparisonfc9_a',
                         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(drop7_a)
    
    
    fc6_b = layers.Dense(1024, activation='relu', name='Comparisonfc6_b', kernel_regularizer=regularizers.l2(l),
                         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(flat_b)
    drop6_b = layers.Dropout(0.5, name='ComparisonDropout6_b')(fc6_b)
    fc7_b = layers.Dense(128, activation='relu', name='Comparisonfc7_b', kernel_regularizer=regularizers.l2(l), 
                         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(drop6_b)
    drop7_b = layers.Dropout(0.5, name='ComparisonDropout7_b')(fc7_b)
    # fc8_b = layers.Dense(128, activation='relu', name='Comparisonfc8_b',  kernel_regularizer=regularizers.l2(l), 
    #                      kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(drop7_b)
    # drop8_b = layers.Dropout(0.1, name='ComparisonDropout8_b')(fc8_b)
    fc9_b = layers.Dense(1, name='Comparisonfc9_b',
                         kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')(drop7_b)



    difference = layers.Subtract(name='Difference')([fc9_a, fc9_b])

    comparison_output = layers.Activation('tanh', name='comparison_output')(difference)


    # summed_difference = layers.Lambda(lambda tensors: tf.reduce_sum(tensors, axis=-1, keepdims=True), name='Summed_Difference')(difference)

    # comparison_output = layers.Dense(1, activation='tanh', name='comparison_output')(summed_difference)

    # # concatenated = layers.Concatenate()([flat_a, flat_b])
    comparison_model = models.Model(inputs=[input_a, input_b], outputs=comparison_output, name='comparison_model')


    return comparison_model