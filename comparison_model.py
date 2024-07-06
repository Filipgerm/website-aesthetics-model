# Comparison model 
from keras import layers, regularizers, models

def create_comparisonModel(shared_base, l=0.001):
    input_shape = shared_base.input_shape[1:]
    
    input_a = layers.Input(shape=input_shape, dtype='float32', name='input_a')
    input_b = layers.Input(shape=input_shape, dtype='float32', name='input_b')

    output_a = shared_base(input_a)
    output_b = shared_base(input_b)

    flat_a = layers.Flatten()(output_a)
    flat_b = layers.Flatten()(output_b)

    # concatenated = layers.Concatenate()([flat_a, flat_b])
    subtracted = layers.Subtract()([flat_a, flat_b])

    fc6 = layers.Dense(1024, activation='relu', name='Comparisonfc6', kernel_regularizer=regularizers.l2(l))(subtracted)
    drop6 = layers.Dropout(0.5, name='ComparisonDropout6')(fc6)
    fc7 = layers.Dense(512, activation='relu', name='Comparisonfc7', kernel_regularizer=regularizers.l2(l))(drop6)
    drop7 = layers.Dropout(0.5, name='ComparisonDropout7')(fc7)

    output = layers.Dense(1, activation='tanh', name='comparison_output')(drop7)

    comparison_model = models.Model(inputs=[input_a, input_b], outputs=output, name='comparison_model')


    return input_a, input_b, comparison_model