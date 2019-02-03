
#importing all dependencies 
import matplotlib 
matplotlib.use("Agg") #using Agg backend to enble saving plots to disk 

from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras import backend as bk
from imutils import paths 
import matplotlib.pyplot as plt 
import numpy as np
import argparse 
import random 
import pickle 
import cv2 as cv
import os 





class SmallVggNet:
	#Only 3x3 convolutions are used
	#Convolution layers are stacked on top of each other deeper in the network architecture prior to applying a destructive pooling 
	def build(height, width, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1 

		if bk.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# #Conv -> Relu -> Pool layer set 
		# #32 conv filters, also the dimensionality of the outpout space 
		# #Filters are 3x3
		# model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim)) #normalize the activations of given input volume before passing it to the next lapyer
		# model.add(MaxPooling2D(pool_size=(2,2))) #reduce the spatial size (width and height). (2,2) will halve the input dimension
		# model.add(Dropout(0.25)) #reduces overfitting

		#(Conv -> Relu)x2 ->  pool layer
		model.add(Conv2D(64, (3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))

		
		model.add(Conv2D(128, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(Conv2D(128, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Conv2D(256, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(Conv2D(256, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(Conv2D(256, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		

		model.add(Conv2D(512, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(Conv2D(512, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(Conv2D(512, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Conv2D(512, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(Conv2D(512, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(Conv2D(512, (3,3), padding="same" ))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis =chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))

		#Fully connected layer -> relu -> ->drouput->softmax
		model.add(Flatten()) #need this as dense layer takes in 1D information 
		model.add(Dense(512)) #output dimension is 512
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(classes, activity_regularizer=l2(0.01)))
		model.add(Activation("softmax"))

		return model






 			


