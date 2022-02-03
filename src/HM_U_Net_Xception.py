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
        inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

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
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        outputs = layers.Conv2D(NUM_CLASSES, 3, activation="softmax", padding="same")(x)

        model = keras.Model(inputs, outputs)
        return model

    def fit(self, hp, model, x, validation_data, callbacks=None, **kwargs):

        with open("params.yaml", 'r') as fd:
            params = yaml.safe_load(fd)
            data_mix = str(params['shelob']['data_mix'])

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
        data_mix = str(params['shelob']['data_mix'])
        epochs = int(params['shelob']['epochs'])
        name = str(params['shelob']['name'])
        test_size = int(params['shelob']['test_size'])
        optim_type = str(params['shelob']['optim_type'])
        learning_rate = float(params['shelob']['learning_rate'])
        trials = int(params['shelob']['trials'])

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
