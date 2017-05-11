
# coding: utf-8

# In[1]:

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


# In[2]:

RANDOM_STATE = 8574
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
DATA_PATH = '/a/data/fisheries_monitoring/data/classifiers/superbox/'
ORIG_PATH = DATA_PATH + 'original'
ORIG_DIST = {}
CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
for cls in  CLASSES:
    files = glob.glob(ORIG_PATH + '/' + cls + '/*.jpg')
    ORIG_DIST[cls] = len(files)
print "Number of examples in each class of the original data:"
for key, val in ORIG_DIST.iteritems():
    print key, '\t', val
ORIG_SIZE = sum(ORIG_DIST.values())
print "Total number of examples in original data:", ORIG_SIZE


# In[3]:

def load_all_labels(folders):
    all_img = []
    all_file_class = []
    for folder in folders:
        img = []
        file_class = []
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
        print "Times of original:", len(file_class)/ORIG_SIZE
        print
        all_img += img
        all_file_class += file_class
    all_labels = pd.DataFrame({"img":all_img, "classes":all_file_class})
    all_labels = all_labels[["img", "classes"]]
    return all_labels



def train_val_labels_split(labels, train_size = 0.8):
    org_labels = labels[labels["img"].str.startswith("original")]
    print "original data size:", len(org_labels)
    train = []
    test = []
    for i in xrange(8):
        train_tmp, test_tmp = train_test_split(org_labels[org_labels["classes"] == i], train_size = train_size, random_state = RANDOM_STATE)
        train.append(train_tmp)
        test.append(test_tmp)
    return pd.concat(train), pd.concat(test)

def aug_train_labels(labels):
    aug = []
    i = 0
    for label in labels["img"]:
        label = label[9:-4]
        aug.append(all_labels[all_labels["img"].str.contains(label)])
        if (i+1) in [k*len(labels)/5 for k in xrange(1,6)]:
            print "Loading augmentated training data...{}% done!".format((i+2)*100/len(labels))
        i += 1
    return pd.concat(aug)

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


# In[4]:

all_folders = glob.glob(DATA_PATH + '*')
all_labels = load_all_labels(all_folders)
print "Total number of examples:", len(all_labels)
print "Total number of times of original:", len(all_labels)/ORIG_SIZE


# In[5]:

train_labels, val_labels = train_val_labels_split(all_labels, train_size = 0.8)
aug_labels = aug_train_labels(train_labels)
train_labels = shuffle(train_labels, random_state = RANDOM_STATE)
val_labels = shuffle(val_labels, random_state = RANDOM_STATE)
aug_labels = shuffle(val_labels, random_state = RANDOM_STATE)
print "original train data size:", len(train_labels)
print "augmented train data size:", len(aug_labels)
print "number of times augmented:", len(aug_labels)/len(train_labels)
print "validation data size:", len(val_labels)


# In[29]:

base_model = ResNet50(weights='imagenet', include_top = False, input_shape=(224,224,3))

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy')


# In[30]:

batch_size = 30
steps_per_epoch = len(aug_labels) / batch_size
nb_epoch = 30
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),]

history = model.fit_generator(generator = data_generator(batch_size, aug_labels, INPUT_WIDTH, INPUT_HEIGHT), 
                              steps_per_epoch = steps_per_epoch,
                              epochs=nb_epoch,
                              verbose=1,
                              callbacks = callbacks,
                              validation_data = data_generator(batch_size, val_labels, INPUT_WIDTH, INPUT_HEIGHT),
                              validation_steps = 30)
model.save('/a/data/fisheries_monitoring/data/models/classifiers/classifier3.h5')


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

