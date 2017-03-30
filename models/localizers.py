import keras.applications.resnet50 as rn50

from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

def ResNet50():
    base_model = rn50.ResNet50(weights='imagenet', include_top = False, input_shape=(224,224,3))

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4)(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model