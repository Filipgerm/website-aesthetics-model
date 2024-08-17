# Comparison model 
from keras import layers, regularizers, models, backend, initializers
import tensorflow as tf
import numpy as np
import random

# Set seeds for reproducibility
seed_value = 400
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
# # os.environ['TF_DETERMINISTIC_OPS'] = '1'

def create_comparisonModel(shared_base, l=0.001):
    input_shape = shared_base.input_shape[1:]
    
    input_a = layers.Input(shape=input_shape, dtype='float32', name='input_a')
    input_b = layers.Input(shape=input_shape, dtype='float32', name='input_b')

    output_a = shared_base(input_a)
    output_b = shared_base(input_b)

    flat_a = layers.Flatten()(output_a)
    flat_b = layers.Flatten()(output_b)
        # flat = layers.Flatten()(shared_base.output)

    # Define Shared Layers
    shared_fc6 = layers.Dense(1024, activation='relu', name='Comparisonfc6', kernel_regularizer=regularizers.l2(l),
                              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')
    shared_drop6 = layers.Dropout(0.5, name='ComparisonDropout6')
    shared_fc7 = layers.Dense(512, activation='relu', name='Comparisonfc7', kernel_regularizer=regularizers.l2(l),
                              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')
    shared_drop7 = layers.Dropout(0.5, name='ComparisonDropout7')
    shared_fc8 = layers.Dense(1, name='Comparisonfc8',
                              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer='zeros')

    # Apply Shared Layers to input_a
    fc6_a = shared_fc6(flat_a)
    drop6_a = shared_drop6(fc6_a)
    fc7_a = shared_fc7(drop6_a)
    drop7_a = shared_drop7(fc7_a)
    fc8_a = shared_fc8(drop7_a)

    # Apply Shared Layers to input_b
    fc6_b = shared_fc6(flat_b)
    drop6_b = shared_drop6(fc6_b)
    fc7_b = shared_fc7(drop6_b)
    drop7_b = shared_drop7(fc7_b)
    fc8_b = shared_fc8(drop7_b)


    difference = layers.Subtract(name='Difference')([fc8_a, fc8_b])

    comparison_output = layers.Activation('tanh', name='comparison_output')(difference)


    # summed_difference = layers.Lambda(lambda tensors: tf.reduce_sum(tensors, axis=-1, keepdims=True), name='Summed_Difference')(difference)

    # comparison_output = layers.Dense(1, activation='tanh', name='comparison_output')(summed_difference)

    # # concatenated = layers.Concatenate()([flat_a, flat_b])
    comparison_model = models.Model(inputs=[input_a, input_b], outputs=comparison_output, name='comparison_model')


    return comparison_model