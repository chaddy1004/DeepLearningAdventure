# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from smallvggnet import SmallVggNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.utils.training_utils import multi_gpu_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2 as cv
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())


data=[]
labels=[]

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(60)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	#load the image, resize it to 64x64 pixel(required input spatrioal dimension of SmallVggNet...apprently)
	#Store the image in the data list 

	image = cv.imread(imagePath)
	image = cv.resize(image,(224,224))
	data.append(image)

	#extract the class label from the image path append to labels list 
	label = imagePath.split(os.path.sep)[-2] 
	#The above operation is same thing as imagePath.split("/"). All it does is it splits the path string with the "/" character, 
	#and you are taking the second last one wihch is the folder name, but also the label
	labels.append(label)

#scaling the RGB values so it resides from 0 to 1 
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)


#use 75% of the data for training and the remiaingin 25% for testing using scikit learn 
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.30, random_state = 42)
print("train size", trainX.shape, "test size", testX.shape)

#convert the labels from integers to vectors for one hot encoding 

lb = LabelBinarizer()
print("Before", trainY) #the labels are still in string form 
lb.fit(trainY) #find the unique class labels in the data
trainY = lb.transform(trainY)
print("After", trainY) #one hot encoding vector 
testY = lb.transform(testY)#same thing done for test data
n_classes = len(lb.classes_)
print(n_classes)
print(lb.classes_)


#construct the image generator for data augumentation 
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

#init the smallvggnet

model = SmallVggNet.build(height=224, width=224, depth=3, classes=len(lb.classes_)) 

INIT_LR = 0.01
EPOCHS = 75
BS = 32

# opt = SGD(lr=INIT_LR, decay=(INIT_LR/EPOCHS))


parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
#train 
# // is integer division in python3
#fit_generator is used instead of fit since our data goes through the data augumentation 
Fit = parallel_model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=(len(trainX)//BS),epochs=EPOCHS)


#evaluate the network 
predictions =  model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

predictions =  parallel_model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, Fit.history["loss"], label="train_loss")
plt.plot(N, Fit.history["val_loss"], label="val_loss")
plt.plot(N, Fit.history["acc"], label="train_acc")
plt.plot(N, Fit.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])


# save the model and label binarizer to disk
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

