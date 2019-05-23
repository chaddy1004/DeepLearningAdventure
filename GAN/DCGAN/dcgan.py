import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam


class GAN():
    def __init__(self):
        # this is the mnist dataset size (28 by 28)
        self.rows = 28
        self.cols = 28
        self.channels = 1
        self.image_shape = (self.rows, self.cols, self.channels)
        self.latent_dim = 102  # input vector for generator 

        opt = Adam(0.0002, 0.8)

        # build and complie discriminator 
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

        # build generator 
        # this returns a keras model
        self.generator = self.build_generator()

        # Takes noise as input and generate images 
        noise_in = Input(shape=(self.latent_dim,))
        # using the model with the keras input tensor 
        image = self.generator(noise_in)

        # only train the generator for combined model
        self.discriminator.trainable = False

        # discriminator takes the generated images as input and determines validity
        validity = self.discriminator(image)

        # combined model: Takes in random noise as input, spits out validity as output 
        # noise---> [Generator] --->  image ---> [Discriminator] ---> validity(percentage)
        # trains the generator to fool the discriminator 
        self.combined = Model(noise_in, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=opt)

    def build_generator(self):
        ###########################
        ##Functional Implementation 
        ############################
        z = Input(shape=(self.latent_dim,))
        dense1 = Dense((7 * 7 * 1024), activation=LeakyReLU(alpha=0.2))(z)
        bn1 = BatchNormalization(momentum=0.8)(dense1)
        reshape = Reshape((7, 7, 1024))(bn1)
        convtrans1 = Conv2DTranspose(512, 5, strides=2, padding="same")(reshape)  # output:7
        lrl1 = LeakyReLU(alpha=0.2)(convtrans1)
        bn2 = BatchNormalization(momentum=0.8)(lrl1)
        convtrans2 = Conv2DTranspose(256, 5, strides=1, padding="same")(bn2)  # output:14
        lrl2 = LeakyReLU(alpha=0.2)(convtrans2)
        bn2 = BatchNormalization(momentum=0.8)(lrl2)
        convtrans3 = Conv2DTranspose(1, 5, strides=2, padding="same")(bn2)  # output:28
        output = Activation("tanh")(convtrans3)
        return Model(z, output)

    def build_discriminator(self):
        ############################
        ##Functional Implementation
        ############################
        image = Input(shape=(self.image_shape))
        conv1 = Conv2D(64, 5, strides=2, padding="same")(image)
        lrl1 = LeakyReLU(alpha=0.2)(conv1)
        bn1 = BatchNormalization(momentum=0.8)(lrl1)
        conv2 = Conv2D(128, 5, strides=2, padding="same")(bn1)
        lrl2 = LeakyReLU(alpha=0.2)(conv2)
        bn2 = BatchNormalization(momentum=0.8)(lrl2)
        flatten = Flatten()(bn2)
        dense = Dense(1024, activation=LeakyReLU(alpha=0.2))(flatten)
        validity = Dense(1, activation="sigmoid")(dense)

        return Model(image, validity)

    def train(self, epochs=100, batch_size=128, sample_interval=50):

        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--generated", required=True, help="path to generated output")
        ap.add_argument("-m", "--model", required=True, help="path to output trained model")
        args = vars(ap.parse_args())

        (X_train, _), (_, _) = mnist.load_data()

        # rescaling to -1 to 1 
        X_train = (X_train / 127.5) - 1
        X_train = np.expand_dims(X_train, axis=3)

        # adversarial ground truths 
        real = np.ones((batch_size, 1))  # probability of 100% for real images
        fake = np.zeros((batch_size, 1))  # probability of 0% for generated images

        for epoch in range(epochs):
            # sample random images from the training set (sampling the the same amount as the batch size)
            i = np.random.randint(0, X_train.shape[0],
                                  batch_size)  # returns array sized batch_size with randomly sampled integers
            images = X_train[i]  # select the corresponding images using the index

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # generate a batch of new images
            image_generated = self.generator.predict(noise)
            # train the discriminator 
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(images, real)
            d_loss_fake = self.discriminator.train_on_batch(image_generated, fake)

            # just taking average of the loss
            d_loss_av = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator 
            self.discriminator.trainable = False
            # creates np array shaped (batch_size, self.latent_dim) that contains values randomly sampled from normal distribution
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # training the combined model to pretend that the generated image is 
            # Since we set the discriminator training to false, this will only train the generator
            g_loss = self.combined.train_on_batch(noise, real)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss_av[0], 100 * d_loss_av[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch, args["generated"])

        self.generator.save(args["model"])

    def sample_images(self, epoch, path):
        row, col = 10, 10

        noise = np.random.normal(0, 1, (row * col, self.latent_dim))
        generated = self.generator.predict(noise)

        # rescale image 
        generated = generated * 255.0

        fig, axis = plt.subplots(row, col)
        count = 0

        for i in range(row):
            for j in range(col):
                axis[i, j].imshow(generated[count, :, :, 0], cmap="gray")
                axis[i, j].axis('off')
                count += 1

        fig.savefig(path + "/images%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=600)
