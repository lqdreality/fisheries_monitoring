import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from keras.utils import np_utils
from keras.preprocessing import image

from scipy.misc import imread
from scipy.misc import imresize


def train_val_test_split(X, y, index, train_len, val_len):
    N = X.shape[0]
    
    idx = np.random.permutation(N)
    idx_train = idx[:int(train_len*N)]
    idx_val = idx[int(train_len*N): int((train_len+val_len)*N)]
    idx_test = idx[int((train_len+val_len)*N): N]
    
    train_X = {'X':X[idx_train], 'y': y[idx_train], 'idx': index[idx_train]}
    val_X = {'X':X[idx_val], 'y': y[idx_val], 'idx': index[idx_val]}
    test_X = {'X':X[idx_test], 'y': y[idx_test], 'idx': index[idx_test]}
    
    return train_X, val_X, test_X

def load_cropped_train(data_path, width, height):
    """
    Loads the dataset of croped images used to train the classifier
    """
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')    
    
    # Run through all the folders where the different classes are stored
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT', 'NoF']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(data_path, fld, '*.jpg')
        files = sorted(glob.glob(path))
        # In every folder, load all the files inside it
        for fl in files:
            flbase = os.path.basename(fl)
            img = image.load_img(fl, target_size=(height, width))
            img = image.img_to_array(img)
            X_train.append(img)
            X_train_id.append(fld + '/' + flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_raw_data(boxes, new_width, new_height, data_path):
    """
    Reads the full dataset and returns all the images, with the annotated boxes and the index of each image. 
    It receives the boxes table with the information of each box and the image that box corresponds to, and 
    the new dimensions (width and height) and returns the images and the boxes. 
    """
    # Initialize some variables
    N = len(boxes)
    raw_data = np.zeros((N, new_height, new_width, 3))
    raw_boxes = np.zeros((N, 4))
    raw_index = []
    
    # Go through all the rows in the boxes table, loading the corresponding image
    for i in xrange(N):
        file_name = boxes.iloc[i]["img"]
        path = data_path + file_name
        img = imread(path)
        box = boxes.iloc[i][["x","y","w","h"]].values.astype(dtype = 'float32')
        
        # Once the image has been loaded, resize it and the box to the desired size
        resized_img, resized_box = resize_data_and_box(img, box, new_width, new_height)
        
        # Store the new values
        raw_data[i] = resized_img
        raw_boxes[i] = box
        raw_index.append(file_name)
        
    return raw_data, raw_boxes, raw_index


def resize_data_and_box(data, box, new_width, new_height):
    """
    Resize an image and its corresponding box
    """
    # Get the old dimensiones
    old_width, old_height = data.shape[0:2]
    old_x, old_y, old_w, old_h = box
    
    # Resize image
    new_img = imresize(data, size=(new_height, new_width))

    # Compute the resized dimensions for the boxs
    new_x = old_x * new_width / old_width
    new_y = old_y * new_height / old_height
    new_w = old_w * new_width / old_width
    new_h = old_h * new_height / old_height
    new_box = np.array([new_x, new_y, new_w, new_h])

    return new_img, new_box


def crop_images(images, boxes, width, height):
    N, old_H, old_W = images.shape[0:3]
    cropped_images = np.zeros((N, height, width, 3))

    for i in range(N):
        x, y, w, h = boxes[i].astype(int)
        
        if (x + w) > old_W:
            hlimit = old_W-1
        else:
            hlimit = x + w
        
        if (y + w) > old_H:
            vlimit = old_H-1
        else:
            vlimit = y + h
            
        if x > old_W:
            x = old_W - 2
        if y > old_H:
            y = old_H - 2
            
        fish = images[i, y:vlimit, x:hlimit, :].copy()
        cropped_images[i] = fish.resize(height, width, 3)
        
    return cropped_images


def visualize_grid(data, preds):
    numrows = 3
    numcols = 3
    
    # Create grid
    _, ax = plt.subplots(numcols, numrows)
    
    # Generate indices of images to show
    idxs = np.random.choice(data.shape[0], size=numcols*numrows, replace=False)
    n = 0
    for i in range(numrows):
        for j in range(numcols):
            idx = idxs[n]
            img = data[n]
            # Visualize it along with the box
            visualize_image(ax, img, i, j, pred_box=preds[idx])
            n+=1
            
    plt.gca().axis('off')
    plt.show()


def visualize_image(ax, img, i, j, true_box = None, pred_box = None):
    ax[i, j].imshow(img)
    
    if true_box is not None:
        x, y, width, height = true_box
        ax[i, j].add_patch(
        patches.Rectangle(
            (x, y), # x,y
            width, # width
            height, # height
            hatch='\\',
            fill=False      # remove background
                )
            )
    if pred_box is not None:
        x, y, width, height = pred_box
        ax[i, j].add_patch(
        patches.Rectangle(
            (x, y), # x,y
            width, # width
            height, # height
            hatch='-',
            fill=False      # remove background
                )
            )