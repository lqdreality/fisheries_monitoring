
# coding: utf-8

# In[86]:

import numpy as np
from scipy.misc import imread
from imgaug import augmenters
# import matplotlib.pyplot as plt
from glob import glob
import os
from PIL import Image


# In[76]:

# Setup some augmenters
aug_dict = {}
aug_dict['Vertical_Flip'] = augmenters.Fliplr(1.0) # Vertical Flip
aug_dict['Horizontal_Flip'] = augmenters.Flipud(1.0) # Horizontal Flip
aug_dict['Blur'] = augmenters.GaussianBlur(5.0) # Blur image
aug_dict['Dropout'] = augmenters.Dropout(0.3, per_channel=True) # Zeros a pixel w/ p=0.3
aug_dict['Add-'] = augmenters.Add(-75, per_channel=True) # Makes it darker
aug_dict['Add+'] = augmenters.Add(75, per_channel=True) # Makes it lighter
aug_dict['Invert'] = augmenters.Invert(0.75, per_channel=True, deterministic=True)
aug_dict['AddGaussianNoise'] = augmenters.AdditiveGaussianNoise(scale=50, per_channel=True) # Adds noise
aug_dict['Emboss'] = augmenters.Emboss(alpha=1.0, strength=1.75)
# Def some affine augmentations too
aug_dict['Rotate+45'] = augmenters.Affine(rotate=45) # Rotates 45 degrees
aug_dict['Rotate-45'] = augmenters.Affine(rotate=-45) # Rotates -45 degrees
aug_dict['Rotate+90'] = augmenters.Affine(rotate=90) # Rotates 90 degrees
aug_dict['Rotate-90'] = augmenters.Affine(rotate=-90) # Rotates -90 degrees
aug_dict['Scale_in'] = augmenters.Affine(scale={"x": 1.5, "y": 1.5}) # Zooms in
aug_dict['Scale_out'] = augmenters.Affine(scale={"x": 0.5, "y": 0.5}) # Zooms out
aug_dict['Translate'] = augmenters.Affine(translate_px={"x": 150, "y": 150})


# In[73]:

CLASSIFIER_DATA_DIR = '/a/data/fisheries_monitoring/data/classifiers/non-superbox/'
CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


# In[78]:

def applyNonAffineAugmentation(dir_name, augmenter, param_name, over_write) :
    for class_i in CLASSES : # for each class
        print '  Class ' + class_i + ' for ' + param_name
        if not os.path.exists(dir_name + '/' + class_i) :
            print dir_name + '/' + class_i + ' does not exist'
            os.mkdir(dir_name + '/' + class_i)
        os.chdir(LOCALIZER_DATA_DIR + 'original/' + class_i)
        images = glob('*.jpg')
        for ff_img in images :
            f_img = ff_img.split('.')[0]
            fname = dir_name + '/' + class_i + '/' + f_img + param_name + '.jpg'
            if os.path.exists(fname) and not over_write :
                continue
            img = np.array(imread(ff_img))
            aug_img = Image.fromarray(augmenter.augment_image(img))
            aug_img.save(fname)


# In[77]:

# Add 90deg, 180deg, 270deg, rotate augmentation
over_write = False
aug_name = 'rotate'
params = [(1,),(2,),(3,)]
dir_name = CLASSIFIER_DATA_DIR + aug_name

if not os.path.exists(dir_name) :
    os.mkdir(dir_name)

for param in params :
    param_name = '_' + str(int(param[0]*90))
    for class_i in CLASSES : # for each class
        print '  Class ' + class_i + ' for ' + param_name
        if not os.path.exists(dir_name + '/' + class_i) :
            print dir_name + '/' + class_i + ' does not exist'
            os.mkdir(dir_name + '/' + class_i)
        os.chdir(CLASSIFIER_DATA_DIR + 'original/' + class_i)
        images = glob('*.jpg')
        for ff_img in images :
            f_img = ff_img.split('.')[0]
            fname = dir_name + '/' + class_i + '/' + f_img + param_name + '.jpg'
            
            img = np.array(imread(ff_img))
            h,w,_ = img.shape
            if not os.path.exists(fname) or over_write : # Rotate and save the images
                aug_img = Image.fromarray(np.rot90(img, param[0]))
                aug_img.save(fname)


# In[79]:

# Add Vertical Flip augmentation
over_write = True
aug_name = 'vflip'
params = [(1.0,)]
dir_name = CLASSIFIER_DATA_DIR + aug_name

if not os.path.exists(dir_name) :
    os.mkdir(dir_name)

for param in params :
    augmenter = augmenters.Fliplr(param[0])
    param_name = '_' + str(int(param[0])) 
    applyNonAffineAugmentation(dir_name, augmenter, param_name, over_write)


# In[123]:

# Add Blur augmentation
over_write = True
aug_name = 'blur'
params = [(2.0,), (5.0,)]
dir_name = CLASSIFIER_DATA_DIR + aug_name

if not os.path.exists(dir_name) :
    os.mkdir(dir_name)

for param in params :
    augmenter = augmenters.GaussianBlur(param[0])
    param_name = '_' + str(int(param[0])) 
    applyNonAffineAugmentation(dir_name, augmenter, param_name, over_write)


# In[124]:

# Add Inversion augmentation
over_write = True
aug_name = 'invert'
params = [(0.15,), (0.75,)]
dir_name = CLASSIFIER_DATA_DIR + aug_name

if not os.path.exists(dir_name) :
    os.mkdir(dir_name)

for param in params :
    augmenter = augmenters.Invert(param[0], per_channel=True, deterministic=True)
    param_name = '_' + str(int(param[0]*100))
    applyNonAffineAugmentation(dir_name, augmenter, param_name, over_write)


# In[131]:

# Add Dropout augmentation
over_write = True
aug_name = 'dropout'
params = [(0.20,), (0.45,)]
dir_name = CLASSIFIER_DATA_DIR + aug_name

if not os.path.exists(dir_name) :
    os.mkdir(dir_name)

for param in params :
    augmenter = augmenters.Dropout(param[0], per_channel=True)
    param_name = '_' + str(int(param[0])) 
    applyNonAffineAugmentation(dir_name, augmenter, param_name, over_write)


# In[127]:

# Add Gaussian Noise augmentation
over_write = True
aug_name = 'gaussianNoise'
params = [(30,), (51,), (80,)]
dir_name = LOCALIZER_DATA_DIR + aug_name

if not os.path.exists(dir_name) :
    os.mkdir(dir_name)

for param in params :
    augmenter = augmenters.AdditiveGaussianNoise(scale=param[0], per_channel=True)
    param_name = '_' + str(int(param[0]))
    applyNonAffineAugmentation(dir_name, augmenter, param_name, over_write)


# In[128]:

# Add Emboss augmentation
over_write = True
aug_name = 'emboss'
params = [(1.0,1.75)]
dir_name = CLASSIFIER_DATA_DIR + aug_name

if not os.path.exists(dir_name) :
    os.mkdir(dir_name)

for param in params :
    augmenter = augmenters.Emboss(alpha=param[0], strength=param[1])
    param_name = '_' + str(int(param[0])) + '_' + str(int(param[1]*100))
    applyNonAffineAugmentation(dir_name, augmenter, param_name, over_write)


# In[129]:

# Add addition augmentation
over_write = True
aug_name = 'add'
params = [(75,),(-75,)]
dir_name = CLASSIFIER_DATA_DIR + aug_name

if not os.path.exists(dir_name) :
    os.mkdir(dir_name)

for param in params :
    augmenter = augmenters.Add(param[0], per_channel=True)
    param_name = '_' + str(int(param[0]))
    applyNonAffineAugmentation(dir_name, augmenter, param_name, over_write)


# ## Count number of files in each folder

# In[117]:

aug_folders = glob(CLASSIFIER_DATA_DIR + '*')
for folder in aug_folders:
    file_names = glob(folder + "/*/*.jpg")
    print "Number of files in folder", folder, ":", len(file_names)


# ## Visualize augmented data

# In[116]:

# files = glob('/a/data/fisheries_monitoring/data/classifiers/superbox/rotate/*/*.jpg')
# n = np.random.choice(len(files))
# one_file = files[n]
# print one_file
# img = np.array(imread(one_file))
# fig = plt.figure(figsize=(25, 25))
# plt.imshow(img)
# plt.show()

