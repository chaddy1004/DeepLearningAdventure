import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from keras.datasets import mnist

from base.base_trainer import BaseTrain


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
        self.model_callbacks = defaultdict(list)
        self.init_callbacks()

    def digit_to_onehot(self, digit_array, batch_size=64):
        one_hot = np.zeros((batch_size, self.config.data.label_len), dtype=np.int8)
        one_hot[np.arange(0, batch_size), digit_array[np.arange(0, batch_size)]] = 1
        return one_hot

    def init_callbacks(self):
        # we want tensorboard to draw the graph of combined model
        self.model_callbacks['combined'].append(
            TensorBoard(log_dir=self.log_dir, batch_size=self.config.trainer.batch_size, write_images=True)
        )
        epochs = self.config.trainer.num_epochs
        steps_per_epoch = self.config.trainer.n_train_data // self.config.trainer.batch_size
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.set_model(model)
                callback.set_params({
                    'batch_size': self.config.trainer.batch_size,
                    'epochs': epochs,
                    'steps': steps_per_epoch,
                    'samples': self.config.trainer.n_train_data,
                    'verbose': True,
                    'do_validation': False,
                    'model_name': model_name
                })

    @staticmethod
    def d_metric_names():
        return ['loss_D', 'loss_D_fake_val', 'loss_D_real_val', 'loss_D_GP']

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

        (X_train, Y_train), (_, _) = mnist.load_data()
        X_train = X_train[0:self.config.trainer.n_train_data, ...]
        Y_train = Y_train[0:self.config.trainer.n_train_data, ...]
        assert self.config.data.label_len == np.max(Y_train + 1)
        X_train = (X_train / 127.5) - 1
        X_train = np.expand_dims(X_train, axis=3)

        train_size = X_train.shape[0]
        steps_per_epoch = train_size // batch_size

        d_metric_names = self.d_metric_names()
        g_metric_names = self.g_metric_names()
        self.on_train_begin()

        for epoch in range(epochs_total):
            self.on_epoch_begin(epoch=epoch, logs={})
            epoch_logs = defaultdict(float)
            for step in range(steps_per_epoch):
                batch_logs = {'batch': step, 'size': self.config.trainer.batch_size}
                metric_logs = defaultdict(float)
                d_metric_logs = defaultdict(float)
                for _ in range(n_critic):
                    i = np.random.randint(0, X_train.shape[0], batch_size)
                    images_real = X_train[i]
                    labels_real = self.digit_to_onehot(Y_train[i], batch_size=batch_size)
                    latent_vector = np.random.normal(0, 1, (batch_size, latent_dim))
                    conditional_latent = np.concatenate((latent_vector, labels_real), axis=-1)
                    images_generated = self.generator.predict(conditional_latent)

                    d_loss = self.discriminator.train_on_batch([images_generated, images_real, labels_real],
                                                               [fake, real, dummy])
                    d_loss = [d_loss] if type(d_loss) != list else d_loss  # In case model only outputs one loss
                    assert (len(d_metric_names) == len(d_loss))  # if loss only has one output, it will not be a list
                    for metric_name, metric_value in zip(d_metric_names, d_loss):
                        d_metric_logs[f'train/{metric_name}'] += metric_value

                for key in d_metric_logs:
                    d_metric_logs[key] /= n_critic

                latent_vector = np.random.normal(0, 1, (batch_size, latent_dim))
                i = np.random.randint(0, self.config.data.label_len, batch_size)
                labels_to_create = self.digit_to_onehot(i, batch_size=batch_size)
                conditional_latent = np.concatenate((latent_vector, labels_to_create), axis=-1)
                g_loss = self.combined.train_on_batch([conditional_latent, labels_to_create], real)
                g_loss = [g_loss] if type(g_loss) != list else g_loss  # In case model only outputs one loss
                assert (len(g_metric_names) == len(g_loss))
                for metric_name, metric_value in zip(g_metric_names, g_loss):
                    metric_logs[f'train/{metric_name}'] += metric_value
                metric_logs.update(d_metric_logs)
                self.print_losses(metric_logs=metric_logs, epoch=epoch, epochs_total=epochs_total, step=step,
                                  steps_per_epoch=steps_per_epoch, start_time=start_time)
                batch_logs.update(metric_logs)

                for metric_name in metric_logs.keys():
                    epoch_logs[metric_name] += metric_logs[metric_name]

                batch_logs = dict(batch_logs)
                self.on_batch_end(batch=step, logs=batch_logs)

            for key in epoch_logs:
                epoch_logs[key] /= steps_per_epoch
            epoch_logs = dict(epoch_logs)
            self.on_epoch_end(epoch=epoch, logs=epoch_logs)
            self.conditional_save_sample(epoch, self.sample_dir)

        self.on_train_end()

    def print_losses(self, metric_logs, epoch, epochs_total, step, steps_per_epoch, start_time):
        print_str = f"[Epoch {epoch + 1}/{epochs_total}] [Batch {step}/{steps_per_epoch}]"
        deliminator = ' '
        for metric_name, metric_value in metric_logs.items():
            if 'accuracy' in metric_name:
                print_str += f"{deliminator}{metric_name}={metric_value:.1f}%"
            elif 'loss' in metric_name:
                print_str += f"{deliminator}{metric_name}={metric_value:.4f}"
            else:
                print_str += f"{deliminator}{metric_name}={metric_value}"
            if deliminator == ' ':
                deliminator = ',\t'

        print_str += f", time: {datetime.datetime.now() - start_time}"
        print(print_str, flush=True)

    # def save_sample(self, epoch, path):
    #     row, col = 10, 10
    #
    #     noise = np.random.normal(0, 1, (row * col, self.config.data.latent_dim))
    #     i = np.random.randint(0, 10, row * col)
    #     labels_to_create = self.digit_to_onehot(i, batch_size=row * col)
    #     conditional_latent = np.concatenate((noise, labels_to_create), axis=-1)
    #     generated = self.generator.predict(conditional_latent)
    #     # rescale image
    #     generated = (generated + 1.) * 127.5
    #
    #     fig, axis = plt.subplots(row, col)
    #     count = 0
    #
    #     for i in range(row):
    #         for j in range(col):
    #             axis[i, j].imshow(np.squeeze(generated[count, ...]), cmap="gray")
    #             axis[i, j].axis('off')
    #             count += 1
    #
    #     fig.savefig(path + "/images%d.png" % epoch)
    #     plt.close()

    def conditional_save_sample(self, epoch, path):
        row, col = self.config.data.label_len, 10
        fig, axis = plt.subplots(row, col)
        for r in range(row):
            latent = np.random.normal(0, 1, (col, self.config.data.latent_dim))
            i = np.ones(col, dtype=np.int8)
            i *= r
            labels_to_create = self.digit_to_onehot(i, batch_size=col)
            conditional_latent = np.concatenate((latent, labels_to_create), axis=-1)
            generated = self.generator.predict(conditional_latent)
            generated = (generated + 1.) * 127.5
            for c in range(col):
                axis[r, c].imshow(np.squeeze(generated[c, ...]), cmap="gray")
                axis[r, c].axis('off')
        fig.savefig(path + "/images%d.png" % epoch)
        plt.close()

    def on_batch_begin(self, batch, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs):
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        logs = logs or {}
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs=None):
        for model_name in self.model_callbacks:
            model = eval(f"self.{model_name}")
            model.stop_training = False
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for model_name in self.model_callbacks:
            callbacks = self.model_callbacks[model_name]
            for callback in callbacks:
                callback.on_train_end(logs)
