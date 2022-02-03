import inference as irnc
import multi_plots as mps
import os
import pandas as pd
import tensorflow as tf
from time import time
import yaml

from tensorflow.keras.callbacks import Callback
from tools import optim_pool
from objects.WeightedCrossEntropy import WeightedCrossEntropy
from objects.BalancedCrossEntropy import BalancedCrossEntropy
from dvclive.keras import DvcLiveCallback
from glob import glob
from tensorflow import keras
from tensorflow.keras import layers
from tools import DATA_DIR, NUM_CLASSES, IMAGE_SIZE, NUM_TRAIN_IMAGES, NUM_VAL_IMAGES
from objects.DataGenerator import DataGenerator


class TimingCallback(Callback):

    def __init__(self):
        super().__init__()
        self.logs = []

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.starttime = time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(time()-self.starttime)


with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    data_mix = str(params['k2000']['data_mix'])
    epochs = int(params['k2000']['epochs'])
    name = str(params['k2000']['name'])
    test_size = int(params['k2000']['test_size'])
    optim_type = str(params['k2000']['optim_type'])
    learning_rate = float(params['k2000']['learning_rate'])
    wce_beta = float(params['dolorean']['wce_beta'])
    bce_beta = float(params['dolorean']['bce_beta'])

train_images = sorted(
    glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/train/**/*.png"), recursive=True))[
               :- test_size]  # [:NUM_TRAIN_IMAGES]
train_masks = sorted(
    glob(os.path.join(DATA_DIR, "finetuning/gtFine/train/**/*octogroups.png"), recursive=True))[
              :- test_size]  # [:NUM_TRAIN_IMAGES]
val_images = sorted(
    glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/val/**/*.png"), recursive=True))  # [:NUM_VAL_IMAGES]
val_masks = sorted(
    glob(os.path.join(DATA_DIR, "finetuning/gtFine/val/**/*octogroups.png"), recursive=True))  # [:NUM_VAL_IMAGES]

print('Found', len(train_images), 'training images')
print('Found', len(train_masks), 'training masks')
print('Found', len(val_images), 'validation images')
print('Found', len(val_masks), 'validation masks')

assert len(train_images) > 0

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

train_dataset = data_blend(train_images, train_masks, data_mix)
val_dataset = data_blend(val_images, val_masks, data_mix)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)


def convolution_block(
        block_input,
        num_filters=256,
        kernel_size=3,
        dilation_rate=1,
        padding="same",
        use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
metrics_wce = WeightedCrossEntropy(beta=wce_beta)
metrics_bce = BalancedCrossEntropy(beta=bce_beta)
optimizer = optim_pool(learning_rate=learning_rate)[optim_type]
cb = TimingCallback()
callback = [DvcLiveCallback(path="./" + name), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3), cb]

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy", metrics_wce, metrics_bce],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[callback])
model.save('models/' + name)

# Time log
df = pd.DataFrame(cb.logs, columns=['time'])
df.index.name = 'index'
df.to_csv(name + '/time.csv', index_label='index')

# Création des plots
mps.main(name)

# Création de la métriques sur jeu de test
irnc.main(name)


