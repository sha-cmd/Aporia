import keras_tuner as kt
import os
import yaml
import tensorflow as tf
import pandas as pd
from objects.DataGenerator import DataGenerator
from glob import glob
from tools import DATA_DIR
from tensorflow.keras import layers
from tensorflow import keras
from tools import IMAGE_SIZE
from tools import NUM_CLASSES


class MyHyperModel(kt.HyperModel):


    def build(self, hp):
        """Builds a convolutional model."""

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

        return DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

    def fit(self, hp, model, x, validation_data, callbacks=None, **kwargs):

        with open("params.yaml", 'r') as fd:
            params = yaml.safe_load(fd)
            data_mix = str(params['stick']['data_mix'])

        batch_size = hp.Int("batch_size", 4, 8, step=2, default=8)

        data_blend = DataGenerator(batch_size)

        train_ds = data_blend(x[0], x[1], data_mix)
        validation_data = data_blend(validation_data[0], validation_data[1], data_mix)
        optim_type = hp.Choice("optim_type", ["adam", "adamax", "ftrl"])
        if optim_type == "adam":
            optimizer = keras.optimizers.Adam(
                hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
            )
        if optim_type == "adamax":
            optimizer = keras.optimizers.Adamax(
                hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
            )
        if optim_type == "ftrl":
            optimizer = keras.optimizers.Ftrl(
                hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
            )

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        epoch_loss_metric = keras.metrics.Mean()

        @tf.function
        def run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        @tf.function
        def run_val_step(images, labels):
            logits = model(images)
            loss = loss_fn(labels, logits)
            epoch_loss_metric.update_state(loss)

        for callback in callbacks:
            callback.model = model

        best_epoch_loss = float("inf")

        for epoch in range(2):
            print(f"Epoch: {epoch}")

            for images, labels in train_ds:
                run_train_step(images, labels)

            for images, labels in validation_data:
                run_val_step(images, labels)

            epoch_loss = float(epoch_loss_metric.result().numpy())
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={"my_metric": epoch_loss})
            epoch_loss_metric.reset_states()

            print(f"Epoch loss: {epoch_loss}")
            best_epoch_loss = min(best_epoch_loss, epoch_loss)

        return best_epoch_loss


def main():
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        data_mix = str(params['stick']['data_mix'])
        epochs = int(params['stick']['epochs'])
        name = str(params['stick']['name'])
        test_size = int(params['stick']['test_size'])
        optim_type = str(params['stick']['optim_type'])
        learning_rate = float(params['stick']['learning_rate'])
        trials = int(params['stick']['trials'])

    keras.backend.clear_session()

    train_images = sorted(
        glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/train/**/*.png"), recursive=True))[:- test_size]
    train_masks = sorted(
        glob(os.path.join(DATA_DIR, "finetuning/gtFine/train/**/*octogroups.png"), recursive=True))[:- test_size]
    val_images = sorted(
        glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/val/**/*.png"), recursive=True))
    val_masks = sorted(
        glob(os.path.join(DATA_DIR, "finetuning/gtFine/val/**/*octogroups.png"), recursive=True))
    print('Found', len(train_images), 'training images')
    print('Found', len(train_masks), 'training masks')
    print('Found', len(val_images), 'validation images')
    print('Found', len(val_masks), 'validation masks')
    tuner = kt.RandomSearch(
        objective=kt.Objective("my_metric", "min"),
        max_trials=trials,
        hypermodel=MyHyperModel(),
        directory="results",
        project_name="custom_training",
        overwrite=True,
    )

    tuner.search(x=(train_images, train_masks), validation_data=(val_images, val_masks))
    print(tuner.search_space_summary())
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps.values)
    df = pd.DataFrame.from_dict({x: [y] for x, y in best_hps.values.items()})
    df.index.name = 'index'
    df.to_csv(name + '/' + name + '_awards.csv', index_label='index')


if __name__ == "__main__":
    main()
