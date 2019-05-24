import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

from base.base_trainer import BaseTrain
from keras.callbacks import TensorBoard


class CombinedTrainer(BaseTrain):
    def __init__(self, generator, discriminator, parallel_discriminator, combined,
                 parallel_combined, config):
        super(CombinedTrainer, self).__init__(config=config)

        self.generator = generator
        self.discriminator = parallel_discriminator
        self.serial_discriminator = discriminator
        self.serial_combined = combined
        self.combined = parallel_combined

        self.log_dir = self.config.exp.log_dir
        self.sample_dir = self.config.exp.sample_dir
        self.info_dir = self.config.exp.info_dir

    def init_callbacks(self):
        self.model_callbacks['combined'].append(
            TensorBoard(log_dir=self.log_dir, batch_size=self.config.trainer.batch_size, write_images=True)
        )


    @staticmethod
    def d_metric_names():
        return ['loss_D', 'loss_D_fake', 'loss_D_real', 'loss_D_GP']

    @staticmethod
    def g_metric_names():
        return ['loss_G']



    def train(self):
        batch_size = self.config.trainer.batch_size
        latent_dim = self.config.data.latent_dim
        real = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))
        n_critic = self.config.trainer.n_critic
        epochs_total = self.config.trainer.num_epochs
        start_time = datetime.datetime.now()

        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train[0:30000,...]
        X_train = (X_train / 127.5) - 1
        X_train = np.expand_dims(X_train, axis=3)
        train_size = X_train.shape[0]
        steps = train_size // batch_size

        d_metric_names = self.d_metric_names()
        g_metric_names = self.g_metric_names()

        for epoch in range(epochs_total):
            epoch_logs = defaultdict(float)
            for step in range(steps):
                metric_logs = {}
                i = np.random.randint(0, X_train.shape[0], batch_size)
                images_real = X_train[i]
                latent_vector = np.random.normal(0, 1, (batch_size, latent_dim))
                images_generated = self.generator.predict(latent_vector)

                d_loss = self.discriminator.train_on_batch([images_generated, images_real], [fake, real, dummy])
                assert (len(d_metric_names) == len(d_loss))
                for metric_name, metric_value in zip(d_metric_names, d_loss):
                    metric_logs[f'train/{metric_name}'] == metric_value

                if step % n_critic == 0:
                    latent_vector = np.random.normal(0, 1, (batch_size, latent_dim))
                    g_loss = self.combined.train_on_batch(latent_vector, real)
                    assert (len(g_metric_names) == len(g_loss))
                    for metric_name, metric_value in zip(d_metric_names, d_loss):
                        metric_logs[f'train/{metric_name}'] == metric_value

                for metric_name in metric_logs.keys():
                    if metric_name in epoch_logs:
                        epoch_logs[metric_name] += metric_logs[metric_name]
                    else:
                        epoch_logs[metric_name] = metric_logs[metric_name]
                print(f"d_loss_real: {d_loss[1]}, d_loss_fake: {d_loss[2]}, d_loss_GP: {d_loss[3]}, g_loss: {g_loss}")
            self.save_sample(epoch, self.log_dir)


    def save_sample(self, epoch, path):
        row, col = 10, 10

        noise = np.random.normal(0, 1, (row * col, self.config.data.latent_dim))
        generated = self.generator.predict(noise)
        # rescale image
        generated = (generated+1.) * 127.5

        fig, axis = plt.subplots(row, col)
        count = 0

        for i in range(row):
            for j in range(col):
                axis[i, j].imshow(generated[count, :, :, 0], cmap="gray")
                axis[i, j].axis('off')
                count += 1

        fig.savefig(path + "/images%d.png" % epoch)
        plt.close()
