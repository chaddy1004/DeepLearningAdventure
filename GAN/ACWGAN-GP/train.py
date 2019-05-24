import json
import os
import random
import shutil

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from model_trainer_builder import build_model_and_trainer


def process_config(json_file, checkpoint=0):
    def get_config_from_json(json_file):
        """
        Get the config from a json file
        :param json_file:
        :return: config(namespace) or config(dictionary)
        """
        # parse the configurations from the config json file provided
        with open(json_file, "r") as config_file:
            config_dict = json.load(config_file)

        # convert the dictionary to a namespace using bunch lib
        config = DotMap(config_dict)

        return config, config_dict

    config, _ = get_config_from_json(json_file)

    # set checkpoint to continue

    # set experiment info
    exp_dir = os.path.join(config.exp.experiment_dir, config.exp.name)
    config.exp.log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(config.exp.log_dir, exist_ok=True)
    config.exp.sample_dir = os.path.join(exp_dir, "samples")
    os.makedirs(config.exp.sample_dir, exist_ok=True)
    config.exp.info_dir = os.path.join(exp_dir, "info")
    os.makedirs(config.exp.info_dir, exist_ok=True)

    # copy the config file
    shutil.copyfile(json_file, os.path.join(exp_dir, os.path.basename(json_file)))

    # set data info1
    config.data.img_shape = (config.data.img_size, config.data.img_size, config.data.img_channels)

    return config


def main():
    config = process_config('configs/config.json')
    print(f"Experiment {config.exp.name}")
    model, trainer = build_model_and_trainer(config)
    print("Start Training")
    trainer.train()


if __name__ == "__main__":
    seed = 19971124
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    main()
