
from tools import IMAGE_SIZE
from tools import BATCH_SIZE
from tools import NUM_CLASSES
from tools import DATA_DIR
from tools import NUM_TRAIN_IMAGES
from tools import NUM_VAL_IMAGES

class ModelVariables:
    def __init__(self):
        print('Variables Init')
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_CLASSES = NUM_CLASSES
        self.DATA_DIR = DATA_DIR
        self.NUM_TRAIN_IMAGES = NUM_TRAIN_IMAGES
        self.NUM_VAL_IMAGES = NUM_VAL_IMAGES

    def __str__(self):
        return f"IMAGE_SIZE = {IMAGE_SIZE}\n"\
                + f"BATCH_SIZE = {BATCH_SIZE}\n"\
                + f"NUM_CLASSES = {NUM_CLASSES}\n"\
                + f"DATA_DIR = {DATA_DIR}\n"\
                + f"NUM_TRAIN_IMAGES = {NUM_TRAIN_IMAGES}\n"\
                + f"NUM_VAL_IMAGES = {NUM_VAL_IMAGES}\n"
