from time import time
from tensorflow.keras.callbacks import Callback


class TimingCallback(Callback):

    def __init__(self):
        super().__init__()
        self.logs = []

    def get_config(self):
        config = super(TimingCallback, self).get_config()
        config['logs'] = []
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.starttime = time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(time()-self.starttime)
