import sys

from .ModelVariables import ModelVariables
from tools import data_original_version
from tools import data_augmented
from tools import BATCH_SIZE

class DataGenerator(ModelVariables):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        print('Eager loading ready')
        self.size_batch = batch_size

    def __call__(self, path_images, path_masks, data_mix='NA'):
        # Security purpose
        assert len(data_mix) <= 16
        assert len(max(path_images, key=len)) <= 256
        assert len(max(path_masks, key=len)) <= 256
        self.path_images = path_images
        self.path_masks = path_masks

        if data_mix == "original_version":
            return self.data_vo()
        elif data_mix == "augmented":
            return self.data_aug()
        else:
            print('paramètre data_mix vaut "augmented" ou "original_version"')
            sys.exit()


    def data_vo(self):
        print('data_vo')
        return data_original_version(self.path_images, self.path_masks, self.size_batch)

    def data_aug(self):
        print('data_aug')
        return data_augmented(self.path_images, self.path_masks, self.size_batch)

