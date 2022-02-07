import numpy as np
import pandas as pd
import tensorflow as tf

IMAGE_SIZE = 128
BATCH_SIZE = 4
NUM_CLASSES = 8
DATA_DIR = "data"
NUM_TRAIN_IMAGES = 16
NUM_VAL_IMAGES = 8


def loss_pool():
    dict_loss = {'binary_cross': tf.keras.losses.binary_crossentropy(),
                 'cat_cross': tf.keras.losses.categorical_crossentropy(),
                 'sparse_cat_cross': tf.keras.losses.sparse_categorical_crossentropy(),
                 'poisson': tf.keras.losses.poisson(),
                 'kl_divergence': tf.keras.losses.kl_divergence()}
    return dict_loss


def optim_pool(learning_rate=0.001):
    dict_optim = {'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  'adadelta': tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
                  'adagrada': tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
                  'adamax': tf.keras.optimizers.Adamax(learning_rate=learning_rate),
                  'nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate),
                  'ftrl': tf.keras.optimizers.Ftrl(learning_rate=learning_rate)}
    return dict_optim


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])

    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255

    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_original_version(image_list, mask_list, batch_size=BATCH_SIZE):
    """Retourne les données telles quelles"""
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def data_augmented(image_list, mask_list, batch_size=BATCH_SIZE):
    """
    Retourne le data generator des données augmentées
    :param image_list: Liste comprenant les chemins vers les images
    :param mask_list: Liste comprenant les chemins vers les masques
    :param batch_size: Taille du batch pour la compilation du modèle
    :return:
    """
    def openImage(image):
         image = tf.io.read_file(image)
         image = tf.image.decode_png(image, channels=0, dtype=tf.dtypes.uint8)
         image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
         image = np.moveaxis(image, (0, 1, 2), (2, 1, 0))
         image = np.array(image)
         return image

    #classes = [openImage(x) for x in mask_list]
    d = {'filename': image_list, 'class': mask_list}
    df = pd.DataFrame(data=d)
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=22)
    data_generator = img_gen.flow_from_dataframe(dataframe=df, color_mode='grayscale',
                                               directory='.', shuffle=False, save_prefix='da',
                                               class_mode='input', batch_size=BATCH_SIZE, save_to_dir='if/')

    return data_generator
