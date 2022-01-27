import tensorflow as tf

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_CLASSES = 8
DATA_DIR = "data"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50


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
        image.set_shape([1024, 2048, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([1024, 2048, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset
