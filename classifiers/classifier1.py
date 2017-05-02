
# coding: utf-8

# In[95]:

import numpy as np
np.random.seed(2016)

import os
import glob
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

from scipy.misc import imread
from scipy.misc import imresize

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import __version__ as keras_version

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


# In[96]:

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
DATA_PATH = '/a/data/fisheries_monitoring/data/classifiers/superbox/'


# In[97]:

aug_folders = glob.glob(DATA_PATH + '*')
for folder in aug_folders:
    print "folder name:", folder


# In[98]:

def load_all_labels(aug_folders):
    img = []
    file_class = []
    for folder in aug_folders:
        folder_name = os.path.basename(folder)
        print('Loading augmentation: {}'.format(folder_name))
        classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
        for idx, cls in enumerate(classes):
            path = os.path.join(DATA_PATH, folder, cls, '*.jpg')
            files = sorted(glob.glob(path))
            for fl in files:
                flbase = os.path.basename(fl)
                img.append(folder_name + '/' + cls + '/' + flbase)
                file_class.append(idx) 
        print "Number of examples:", len(file_class)
        print 
    all_labels = pd.DataFrame({"img":img, "classes":file_class})
    all_labels = all_labels[["img", "classes"]]
    return all_labels


def train_val_test_split(all_labels, val_size, test_size):
    all_labels = shuffle(all_labels, random_state = 8574)
    test_labels = all_labels[0:test_size]
    val_labels = all_labels[test_size:test_size + val_size]
    train_labels = all_labels[test_size + val_size:]
    return train_labels, val_labels, test_labels


def data_generator(batch_size, labels, INPUT_WIDTH, INPUT_HEIGHT):
    while True:
        img_batch = np.zeros((batch_size, INPUT_WIDTH, INPUT_HEIGHT, 3))
        class_batch = np.zeros((batch_size, 8))
        for i in xrange(batch_size):
            n = np.random.choice(len(labels))
            file_name = labels.iloc[n]["img"]
            path = DATA_PATH + file_name
            img = image.load_img(path)
            img = img.resize((INPUT_WIDTH, INPUT_HEIGHT))
            img = image.img_to_array(img)
            img /= 255
            img_batch[i] = img
            
            class_batch[i] = np_utils.to_categorical(labels.iloc[n]["classes"], 8)
        
        yield (img_batch, class_batch)

def load_data(labels, INPUT_WIDTH, INPUT_HEIGHT):
    X = []
    y = []
    idx = []
    X_raw = []
    y_raw = [] 
    shape_raw = []
    for i in xrange(len(labels)):
        file_name = labels.iloc[i]["img"]
        path = DATA_PATH + file_name
        img_raw = image.load_img(path)
        img = img_raw.resize((INPUT_WIDTH,INPUT_HEIGHT))
        
        img_raw = image.img_to_array(img_raw)
        img_raw /= 255
        img = image.img_to_array(img)
        img /= 255
        
        file_class = labels.iloc[i]["classes"]

        X.append(img)
        y.append(file_class)
        idx.append(file_name)
        X_raw.append(img_raw)
        
        if (i+1) in [k*len(labels)/5 for k in xrange(1,6)]:
                print "Loading...{}% done!".format((i+2)*100/len(labels))
        
    return np.array(X), np.array(y), np.array(idx), np.array(X_raw)


def visualize_prediction(img, index = None, true_box = None, pred_box = None, ax = None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12,8))
    ax.imshow(img)
    if index is not None:
        ax.set_title(index)
    height = img.shape[0]
    width = img.shape[1]
    
    if true_box is not None:
        x, y, w, h = true_box
        x = x * width
        y = y * height
        w = w * width
        h = h * height
        ax.add_patch(
        patches.Rectangle(
            (x, y), # x,y
            w, # width
            h, # height
            hatch='\\',
            fill=False,      # remove background
            color = 'r',
            linewidth = 2.5
                )
            )
    if pred_box is not None:
        x, y, w, h = pred_box
        x = x * width
        y = y * height
        w = w * width
        h = h * height
        ax.add_patch(
        patches.Rectangle(
            (x, y), # x,y
            w, # width
            h, # height
            hatch='-',
            fill=False,      # remove background
            color = 'k',
            linewidth = 2.5
                )
            )

def make_plot(data, nrow = 2, ncol = 2, index = None, true_box = None, pred_box = None, figsize = (15,8)):
    # Create grid
    _, ax = plt.subplots(nrow, ncol, figsize=figsize)
    
    idx = None
    tbox = None
    pbox = None
    # Generate indices of images to show
    for axi in np.ravel(ax):
        n = np.random.choice(len(data))
        img = data[n]
        if index is not None:
            idx = index[n]
        if true_box is not None:
            tbox = true_box[n]
        if pred_box is not None:
            pbox = pred_box[n]
        
        # Visualize it along with the box
        visualize_prediction(img, index = idx, true_box = tbox, pred_box = pbox, ax = axi)


# In[99]:

all_labels = load_all_labels(aug_folders)


# In[100]:

selected_aug = {"ALB" : ["original"],
                "BET" : ["original"],
                "DOL" : ["original"],
                "LAG" : ["original"],
                "NoF" : ["original"],
                "OTHER" : ["original"],
                "SHARK" : ["original"],
                "YFT" : ["original"]}
selected_labels = pd.DataFrame(columns = ["img", "classes"])
for key, values in selected_aug.iteritems():
    for value in values:
        labels_tmp = all_labels[all_labels["img"].str.startswith(value + '/' + key)]
        selected_labels = selected_labels.append(labels_tmp)
selected_labels["classes"] = selected_labels["classes"].astype(int)


# In[101]:

train_labels, val_labels, test_labels = train_val_test_split(selected_labels, int(0.2*len(selected_labels)), 0)
print "all data size: ", len(selected_labels)
print "train data size:", len(train_labels)
print "validation data size:", len(val_labels)
print "test data size:", len(test_labels)


# In[102]:

base_model = ResNet50(weights='imagenet', include_top = False, input_shape=(224,224,3))

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy')


# In[ ]:

batch_size = 30
steps_per_epoch = len(train_labels) / batch_size
nb_epoch = 30
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),]

history = model.fit_generator(generator = data_generator(batch_size, train_labels, INPUT_WIDTH, INPUT_HEIGHT), 
                              steps_per_epoch = steps_per_epoch,
                              epochs=nb_epoch,
                              verbose=1,
#                               callbacks = callbacks,
                              validation_data = data_generator(batch_size, val_labels, INPUT_WIDTH, INPUT_HEIGHT),
                              validation_steps = 30)
model.save('/a/data/fisheries_monitoring/data/models/classifiers/classifier1.h5')


# In[ ]:

X_test, y_test, id_test, X_test_raw = load_data(val_labels, INPUT_WIDTH, INPUT_HEIGHT)
predictions_valid = model.predict(X_test.astype('float32'), batch_size=batch_size, verbose=1)
score = log_loss(y_test, predictions_valid)
print "log loss score: ", score


# In[ ]:

from sklearn.metrics import accuracy_score
y_pred = np.argmax(predictions_valid, axis = 1)
acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print "accuracy: ", acc


# In[ ]:

print y_test[0:35]
print y_pred[0:35]

print y_test[35:70]
print y_pred[35:70]

