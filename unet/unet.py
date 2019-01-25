#importing all dependencies 
import matplotlib 
matplotlib.use("Agg") #using Agg backend to enble saving plots to disk 

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout 
from keras.layers import BatchNormalization, Activation, ZeroPadding2D 
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,  Conv2D, Conv2DTranspose, MaxPooling2D 
from keras.models import Sequential, Model
from keras.optimizers import Adam
import argparse






class UNET:
	
	def __init__(self, imgRows=572, imgCols=572, imgChannels=1):
		self.rows = imgRows
		self.cols = imgCols 
		self.channels = imgChannels 
		self.image_shape = (self.rows, self.cols, self.channels)

		self.network = self.build()
		opt = Adam(0.0002, 0.8)
		self.network.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

	def two_conv3x3(self,layer_input, filters = 64):
		conv1 = Conv2D(filters,3,activation="relu")(layer_input)
		conv2 = Conv2D(filters,3,activation="relu")(conv1)

		return conv2 

	#layer building for the "encoder" part
	def build_input(self):
		input_layer = Input(shape=(self.image_shape))
		return input_layer


	def build_layer1(self, layer_input):
		layer1_out = self.two_conv3x3(layer_input)
		return layer1_out


	def build_layer2(self, layer_input):
		pooling = MaxPooling2D()(layer_input)
		layer2_out = self.two_conv3x3(pooling, 128)
		return layer2_out


	def build_layer3(self, layer_input):
		pooling = MaxPooling2D()(layer_input)
		layer3_out = self.two_conv3x3(pooling, 256)
		return layer3_out


	def build_layer4(self, layer_input):
		pooling = MaxPooling2D()(layer_input)
		layer4_out = self.two_conv3x3(pooling, 512)
		return layer4_out


	def build_lastlayer(self, layer_input):
		pooling = MaxPooling2D()(layer_input)
		lastlayer_out = self.two_conv3x3(pooling, 1024)
		return lastlayer_out


	#layer building for the "decoder" part, indicated by the _up tag from the word upsampling 
	def build_layer1_up(self, layer_input):
		





	def build(self):
		input_layer = self.build_input()
		layer1 = self.build_layer1(input_layer)
		
		layer2 = self.build_layer2(layer1)
		
		layer3 = self.build_layer3(layer2)

		layer4 = self.build_layer4(layer3)

		layer_last = self.build_lastlayer(layer4)

		model = Model(input_layer, layer_last)
		model.summary()

		return model


if __name__=='__main__':
	unet = UNET()
	



