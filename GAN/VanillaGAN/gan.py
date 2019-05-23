import argparse

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
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

        # combinedmodel: Takes in random noise as input, spits out validity as output
        # noise---> [Generator] --->  image ---> [Discriminator] ---> validity(percentage)
        # trains the generator to fool the discriminator
        self.combined = Model(noise_in, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=opt)

    def build_generator(self):
        ###########################
        ##Sequential Implementation
        ###########################
        # model = Sequential()
        # model.add(Dense(256, input_dim = self.latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(np.prod(self.image_shape), activation = 'tanh'))
        # model.add(Reshape(self.image_shape))
        # model.summary()
        # noise=Input(shape=(self.latent_dim))
        # image = model(noise)

        ###########################
        ##Functional Implementation
        ############################
        z = Input(shape=(self.latent_dim,))
        x = Dense(256)(z)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(np.prod(self.image_shape), activation='tanh')(x)
        image = Reshape(self.image_shape)(x)
        model = Model(z, image)
        model.summary()

        # input is noise, image is the output
        return Model(z, image)

    def build_discriminator(self):
        # ############################
        # ##Sequential Implementation
        # ############################
        # model = Sequential()
        # model.add(Flatten(input_shape=self.image_shape))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha =0.2))
        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1, activation='sigmoid')) #meaning that the output is a single number
        # model.summary()
        # image = Input(shape=self.image_shape)
        # validity = model(image)

        ############################
        ##Functional Implementation
        ############################
        image = Input(shape=(self.image_shape))
        x = Flatten()(image)
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
        validity = Dense(1, activation='sigmoid')(x)

        return Model(image, validity)

    def train(self, epochs=100, batch_size=128, sample_interval=50):

        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--generated", required=True, help="path to generated output")
        ap.add_argument("-m", "--model", required=True, help="path to output trained model")
        args = vars(ap.parse_args())

        (X_train, _), (_, _) = mnist.load_data()

        # rescaling to -1 to 1
        X_train = X_train / 255.0
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
