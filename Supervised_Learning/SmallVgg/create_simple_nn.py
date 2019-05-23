#!/usr/bin/env python
# coding: utf-8

# # Keras Tutorial Using Dog, Cat, and Panda Images (based on pyimagesearch.com)
# 

# ## Preprocessing 

# ### Imports

# In[1]:


# importing all dependencies
import matplotlib

matplotlib.use("Agg")  # using Agg backend to enble saving plots to disk

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as bk
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2 as cv
import os

# checking to see if it is using gpu backend
bk.tensorflow_backend._get_available_gpus()

# In[2]:


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True, help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# ### Importing and Resizing the Images

# In[3]:


data = []
labels = []
pixel_new = 32

imagePaths = sorted(list(paths.list_images(args["dataset"])))

random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv.imread(imagePath)
    # resize the images into 32x32 pixel image with the aspect ratio ignored.
    # Then each image is flattened into 32x32x3 pixel image
    image = cv.resize(image, (pixel_new, pixel_new)).flatten()
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

print("length of labels", len(labels))  # should be 3000: 1000 along 3 animal types
print("length of data", len(data[1]))

# In[4]:


data = np.array(data, dtype="float") / 255.0  # normalizing the pixel data to go from [0,1] instead of [0,255]
labels = np.array(labels)

print(data.shape, labels.shape)

# ## Creating Training and Testing Data

# In[5]:


# use 75% of the data for training and the remiaingin 25% for testing using scikit learn
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
print("train size", trainX.shape, "test size", testX.shape)

# In[6]:


# convert the labels from integers to vectors for one hot encoding

lb = LabelBinarizer()
print("Before", trainY)  # the labels are still in string form
lb.fit(trainY)  # find the unique class labels in the data
trainY = lb.transform(trainY)  # transforms the labels to one hot encoding using the  information from fit
print("After", trainY)  # one hot encoding vector
testY = lb.transform(testY)  # same thing done for test data
n_classes = len(lb.classes_)
print(n_classes)
print(lb.classes_)

# ## Creating Network

# Using one input layer, two hidden layers, and one output layers 

# In[7]:


model = Sequential()
inputDim = pixel_new * pixel_new * 3
# the data is vector with size 3072 (32x32x3)
model.add(Dense(1024, input_dim=inputDim,
                activation="sigmoid"))  # only need to specify the input dimension to the first layers
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(n_classes, activation="softmax"))

# ### Traning the Created Network

# In[8]:


INIT_LR = 0.01
EPOCHS = 75

opt = SGD(lr=INIT_LR)  # define stochastic gradient dsecent as the optimizer with the initial learning rate
# using categorial crossentropy as the loss fucntion. Use accuracy as metrics
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# In[11]:


# train the newural network

print("Traning Network...")
# Epoch:When the entire dataset  is passed forward and backward through the nerural network once
# batch size:  Total number of training examples present in a single batch(dividing the dataset into batches)
Fit = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# ### Making Prediction with the Trained Network  using Test Data

# In[12]:


predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.figure()
plt.plot(N, Fit.history["loss"], label="train_loss")
plt.plot(N, Fit.history["val_loss"], label="val_loss")
plt.plot(N, Fit.history["acc"], label="train_acc")
plt.plot(N, Fit.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
