# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages

from smallvggnet import SmallVggNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# use 75% of the data for training and the remiaingin 25% for testing using scikit learn
(trainX, trainY), (testX, testY) = cifar10.load_data()
print("train size", trainX.shape, "test size", testX.shape)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255
# convert the labels from integers to vectors for one hot encoding

lb = LabelBinarizer()
# print("Before", trainY) #the labels are still in string form 
lb.fit(trainY)  # find the unique class labels in the data
print("After", trainY)  # one hot encoding vector
trainY = lb.transform(trainY)
testY = lb.transform(testY)  # same thing done for test data
n_classes = len(lb.classes_)
print(n_classes)
print(lb.classes_)
classes = list(map(str, lb.classes_))
print(classes)
print(type(classes[0]))

# construct the image generator for data augumentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# init the smallvggnet

model = SmallVggNet.build(height=32, width=32, depth=3, classes=10)

INIT_LR = 0.01
EPOCHS = 50
BS = 128

# opt = SGD(lr=INIT_LR, decay=(INIT_LR/EPOCHS))


parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
# train
# // is integer division in python3
# fit_generator is used instead of fit since our data goes through the data augumentation
Fit = parallel_model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
                                   steps_per_epoch=(len(trainX) // BS), epochs=EPOCHS)

# evaluate the network
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classes))

predictions = parallel_model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classes))

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
