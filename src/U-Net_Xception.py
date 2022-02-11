import os
import yaml
import multi_plots as mps
import pandas as pd
import inference as irnc
import tensorflow as tf
import re
import sys

from objects.WeightedCrossEntropy import WeightedCrossEntropy
from objects.BalancedCrossEntropy import BalancedCrossEntropy
from objects.DataGenerator import DataGenerator
from glob import glob
from tools import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES, NUM_VAL_IMAGES, NUM_TRAIN_IMAGES
from tensorflow.keras import layers
from tensorflow import keras
from dvclive.keras import DvcLiveCallback
from random import sample
from time import time
from tools import integrate

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    data_mix = str(params['dolorean']['data_mix'])
    epochs = int(params['dolorean']['epochs'])
    name = str(params['dolorean']['name'])
    test_size = int(params['dolorean']['test_size'])
    optim_type = str(params['dolorean']['optim_type'])
    learning_rate = float(params['dolorean']['learning_rate'])


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))
    inputs = tf.image.grayscale_to_rgb(
        inputs, name=None
    )
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


keras.backend.clear_session()

model = get_model((IMAGE_SIZE, IMAGE_SIZE), NUM_CLASSES)

if data_mix == 'original_version':
    train_images_str = DATA_DIR + "/coarse_tuning/leftImg8bit/train/**/*.png"
    train_masks_str = DATA_DIR + "/finetuning/gtFine/train/**/*octogroups.png"
    val_images_str = DATA_DIR + "/coarse_tuning/leftImg8bit/val/**/*.png"
    val_masks_str = DATA_DIR + "/finetuning/gtFine/val/**/*octogroups.png"
elif data_mix == 'multiplication':
    train_images_str = DATA_DIR + "/coarse_tuning/leftImg8bit/train/**/*.*g"
    train_masks_str = DATA_DIR + "/finetuning/gtFine/train/**/*octogroups.*g"
    val_images_str = DATA_DIR + "/coarse_tuning/leftImg8bit/val/**/*.*g"
    val_masks_str = DATA_DIR + "/finetuning/gtFine/val/**/*octogroups.*g"
else:
    sys.exit()

train_images = sorted(
    glob(os.path.join(train_images_str), recursive=True))[:NUM_TRAIN_IMAGES]
train_masks = sorted(
    glob(os.path.join(train_masks_str), recursive=True))[:NUM_TRAIN_IMAGES]
val_images = sorted(
    glob(os.path.join(val_images_str), recursive=True))[:NUM_VAL_IMAGES]
val_masks = sorted(
    glob(os.path.join(val_masks_str), recursive=True))[:NUM_VAL_IMAGES]

test_images = []
test_masks = []

# Cette ligne doit-être commenté durant le développement
# pour ne pas faire de trop long apprentissage chronophage
NUM_TRAIN_IMAGES = len(train_images)

# Selection des données de test
indextestlist = sorted(sample([x for x in range(NUM_TRAIN_IMAGES)], test_size))[::-1]
indextrainlist = [x for x in range(NUM_TRAIN_IMAGES)]
test_images = [train_images[i] for i in indextestlist]
test_masks = [train_masks[i] for i in indextestlist]
# Suppression des doublons dans les données d’entraînement
for i in indextestlist:
    del indextrainlist[i]
train_images = [train_images[i] for i in indextrainlist]
train_masks = [train_masks[i] for i in indextrainlist]

print('Found', len(train_images), 'training images')
print('Found', len(train_masks), 'training masks')
print('Found', len(val_images), 'validation images')
print('Found', len(val_masks), 'validation masks')
print('Found', len(test_images), 'test images')
print('Found', len(test_masks), 'test masks')


for i in range(len(train_images)):
    assert train_images[i].split(
        '/')[-1].split('_leftImg8bit')[0] == train_masks[i].split('/')[-1] \
               .split('_gtFine_polygons_octogroups')[0]
print('Train images correspond to train masks')

for i in range(len(val_images)):
    assert val_images[i].split('/')[-1].split('_leftImg8bit')[0] == val_masks[i] \
        .split('/')[-1].split('_gtFine_polygons_octogroups')[0]
print('Validation images correspond to validation masks')


data_blend = DataGenerator()
data_blend_val = DataGenerator()
train_dataset = data_blend(train_images, train_masks, data_mix)
val_dataset = data_blend_val(val_images, val_masks, data_mix)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)


loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics_wce = WeightedCrossEntropy
metrics_bce = BalancedCrossEntropy

callbacks = [DvcLiveCallback(path="./" + name), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy", metrics_wce, metrics_bce])
start_time = time()
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, batch_size=BATCH_SIZE, callbacks=callbacks)
model.save('models/' + name)

# Time log
time = round((time()-start_time), 2)

# Création des plots
mps.main(name)

# Création de la métriques sur jeu de test
mIoU, dice = irnc.main(test_images, test_masks, name)

metric = [time, mIoU, dice]
metric_name = ['time', 'mIoU', 'dice']
# Time log Intégration au tableau de comparatif DVC
# mIoU Integration au tableau de comparatif DVC
# Dice Integration au tableau de comparatif DVC
integrate(metric, metric_name, name)




