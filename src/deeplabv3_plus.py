import os
import tensorflow as tf
import yaml
import multi_plots as mps
import inference as irnc

from dvclive.keras import DvcLiveCallback
from glob import glob
from tensorflow import keras
from tensorflow.keras import layers
from tools import DATA_DIR, NUM_CLASSES, IMAGE_SIZE, NUM_TRAIN_IMAGES, NUM_VAL_IMAGES
from objects.DataGenerator import DataGenerator

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    data_mix = str(params['k2000']['data_mix'])
    epochs = int(params['k2000']['epochs'])

train_images = sorted(
    glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/train/**/*.png"), recursive=True))#[:NUM_TRAIN_IMAGES]
train_masks = sorted(
    glob(os.path.join(DATA_DIR, "finetuning/gtFine/train/**/*octogroups.png"), recursive=True))#[:NUM_TRAIN_IMAGES]
val_images = sorted(
    glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/val/**/*.png"), recursive=True))#[:NUM_VAL_IMAGES]
val_masks = sorted(
    glob(os.path.join(DATA_DIR, "finetuning/gtFine/val/**/*octogroups.png"), recursive=True))#[:NUM_VAL_IMAGES]

print('Found', len(train_images), 'training images')
print('Found', len(train_masks), 'training masks')
print('Found', len(val_images), 'validation images')
print('Found', len(val_masks), 'validation masks')

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

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
callback = [DvcLiveCallback(path="./k2000"), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=["accuracy"],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[callback])
#model.save('models/k2000')

# Création des plots
mps.main()

# Création de la métriques sur jeu de test
irnc.main()
