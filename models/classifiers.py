from keras.layers.core import Dense, Flatten
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


def InceptionV3():
    import keras.applications.inception_v3 as iv3
    # create the base pre-trained model
    base_model = iv3.InceptionV3(weights='imagenet', include_top=False)
    
    return pretrained_affine_classifier(base_model, 1024, 8)


def ResNet50():
    import keras.applications.resnet50 as rn50
    base_model = rn50.ResNet50(weights='imagenet', include_top = False, input_shape=(224,224,3))

    return pretrained_affine_classifier(base_model, 1024, 8)
    

def pretrained_affine_classifier(base_model, hidden_dim, num_output, opt='sgd', loss='categorical_crossentropy'):
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hidden_dim, activation='relu')(x)
    predictions = Dense(num_output, activation='softmax')(x)
    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(opt, loss)
    
    return model