from tensorflow import keras
from keras import layers, models, regularizers
from lrn_layer import LRN
import numpy as np

class SharedFeatureExtractor:
    def __init__(self, input_shape=(192, 256, 3), l=0.001):
        self.input_shape = input_shape
        self.l = l

    def build_model(self, layer_config):
        im_data = layers.Input(shape=self.input_shape, dtype='float32', name='im_data')
        x = im_data

        for layer in layer_config:
            layer_type = layer.pop('type')  # Remove and get the 'type' key
            if layer_type == 'Conv2D':
                x = self._add_conv2d_layer(x, **layer)
            elif layer_type == 'MaxPooling2D':
                x = self._add_maxpooling2d_layer(x, **layer)
            elif layer_type == 'LRN':
                x = self._add_lrn_layer(x, **layer)
            elif layer_type == 'Dropout':
                x = self._add_dropout_layer(x, **layer)
            elif layer_type == 'Split':
                x = self._add_split_layers(x, **layer)
            elif layer_type == 'Concatenate':
                x = self._add_concatenate_layer(x, **layer)
            layer['type'] = layer_type  # Add the 'type' key back

        shared_model = models.Model(inputs=im_data, outputs=x, name='shared_feature_extractor')
        return shared_model

    def _add_conv2d_layer(self, x, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', name=None, **kwargs):
        if isinstance(x, list):
            x = layers.Concatenate()(x)  # Concatenate if multiple inputs
        return layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=activation,
                             padding=padding, name=name, kernel_regularizer=regularizers.l2(self.l))(x)

    def _add_maxpooling2d_layer(self, x, pool_size=(3, 3), strides=(2, 2), padding="same", name=None):
        if isinstance(x, list):
            x = layers.Concatenate()(x)  # Concatenate if multiple inputs
        return layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)(x)

    def _add_lrn_layer(self, x, name=None):
        if isinstance(x, list):
            x = layers.Concatenate()(x)  # Concatenate if multiple inputs
        return LRN(name=name)(x)

    def _add_dropout_layer(self, x, rate, name=None):
        if isinstance(x, list):
            x = layers.Concatenate()(x)  # Concatenate if multiple inputs
        return layers.Dropout(rate, name=name)(x)

    def _add_split_layers(self, x, split_index, names):
        layer1 = layers.Lambda(lambda x: x[:, :, :, :split_index], name=names[0])(x)
        layer2 = layers.Lambda(lambda x: x[:, :, :, split_index:], name=names[1])(x)
        return [layer1, layer2]

    def _add_concatenate_layer(self, x, name=None):
        if isinstance(x, list) and len(x) > 1:
            return layers.Concatenate(name=name)(x)
        elif isinstance(x, list):
            return x[0]  # Return the single element
        return x
    
    # Add this method to perform a test forward pass
    def test_forward_pass(self, model):
        # Create a random input image with the correct dimensions
        random_image = np.random.rand(1, *self.input_shape).astype(np.float32)
        
        # Perform a forward pass
        output = model.predict(random_image)
        
        print(f"Input shape: {random_image.shape}")
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!")

if __name__ == "__main__":
    # Example usage
    layer_config = [
    {'type': 'Conv2D', 'filters': 96, 'kernel_size': (11, 11), 'strides': (4, 4), 'activation': 'relu', 'padding': 'valid', 'name': 'conv1'},
    {'type': 'MaxPooling2D', 'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'same', 'name': 'pool1'},
    {'type': 'LRN', 'name': 'norm1'},
    {'type': 'Dropout', 'rate': 0.1, 'name': 'dropout1'},
    {'type': 'Split', 'split_index': 48, 'names': ['split1_1', 'split1_2']},
    {'type': 'Conv2D', 'filters': 128, 'kernel_size': (5, 5), 'activation': 'relu', 'padding': 'same', 'name': 'conv2_1'},
    {'type': 'Conv2D', 'filters': 128, 'kernel_size': (5, 5), 'activation': 'relu', 'padding': 'same', 'name': 'conv2_2'},
    {'type': 'Concatenate', 'name': 'conv_2'},
    {'type': 'MaxPooling2D', 'pool_size': (3, 3), 'strides': (2, 2), 'name': 'pool2'},
    {'type': 'LRN', 'name': 'norm2'},
    {'type': 'Dropout', 'rate': 0.1, 'name': 'dropout2'},
    {'type': 'Conv2D', 'filters': 384, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same', 'name': 'conv3'},
    {'type': 'Dropout', 'rate': 0.1, 'name': 'dropout3'},
    {'type': 'Split', 'split_index': 192, 'names': ['split3_1', 'split3_2']},
    {'type': 'Conv2D', 'filters': 192, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same', 'name': 'conv4_1'},
    {'type': 'Conv2D', 'filters': 192, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same', 'name': 'conv4_2'},
    {'type': 'Concatenate', 'name': 'conv_4'},
    {'type': 'Split', 'split_index': 192, 'names': ['split4_1', 'split4_2']},
    {'type': 'Conv2D', 'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same', 'name': 'conv5_1'},
    {'type': 'Conv2D', 'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same', 'name': 'conv5_2'},
    {'type': 'Concatenate', 'name': 'conv_5'},
    {'type': 'MaxPooling2D', 'pool_size': (3, 3), 'strides': (2, 2), 'name': 'pool5'},
]

    feature_extractor = SharedFeatureExtractor()
    model = feature_extractor.build_model(layer_config)
    model.summary()
    print("\nPerforming Test forward pass: ")
    feature_extractor.test_forward_pass(model)
