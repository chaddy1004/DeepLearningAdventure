import abc


class BaseTrain(object):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
