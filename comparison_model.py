# Comparison model 
from keras import layers, regularizers, models, backend

def create_comparisonModel(shared_base, l=0.001):
    input_shape = shared_base.input_shape[1:]
    
    input_a = layers.Input(shape=input_shape, dtype='float32', name='input_a')
    input_b = layers.Input(shape=input_shape, dtype='float32', name='input_b')

    output_a = shared_base(input_a)
    output_b = shared_base(input_b)

    flat_a = layers.Flatten()(output_a)
    flat_b = layers.Flatten()(output_b)


    fc6_a = layers.Dense(1024, activation='relu', name='Comparisonfc6_a', kernel_regularizer=regularizers.l2(l))(flat_a)
    drop6_a = layers.Dropout(0.5, name='ComparisonDropout6_a')(fc6_a)
    fc7_a = layers.Dense(512, activation='relu', name='Comparisonfc7_a', kernel_regularizer=regularizers.l2(l))(drop6_a)
    drop7_a = layers.Dropout(0.5, name='ComparisonDropout7_a')(fc7_a)
    # embedding_a = layers.Dense(256, activation=None, name='embedding_a')(drop7_a)

    fc6_b = layers.Dense(1024, activation='relu', name='Comparisonfc6_b', kernel_regularizer=regularizers.l2(l))(flat_b)
    drop6_b = layers.Dropout(0.5, name='ComparisonDropout6_b')(fc6_b)
    fc7_b = layers.Dense(512, activation='relu', name='Comparisonfc7_b', kernel_regularizer=regularizers.l2(l))(drop6_b)
    drop7_b = layers.Dropout(0.5, name='ComparisonDropout7_b')(fc7_b)
    # embedding_b = layers.Dense(256, activation=None, name='embedding_b')(drop7_b)    

    # Use L1 distance for comparison
    difference = layers.Subtract(name='Difference')([drop6_a, drop6_b])
    comparison_output = layers.Dense(1, activation='tanh', name='comparison_output')(difference)

    # # concatenated = layers.Concatenate()([flat_a, flat_b])



    comparison_model = models.Model(inputs=[input_a, input_b], outputs=comparison_output, name='comparison_model')


    return input_a, input_b, comparison_model