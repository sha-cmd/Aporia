import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os

from scipy.io import loadmat
from tensorflow import keras
from tools import read_image
from glob import glob
from tools import DATA_DIR

test_images = sorted(glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/test/**/*.png"), recursive=True))
test_masks = sorted(glob(os.path.join(DATA_DIR, "finetuning/gtFine/test/**/*octogroups.png"), recursive=True))

history = keras.models.load_model('models/keras_model')



# Loading the Colormap
colormap = loadmat(
    "src/city_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()


def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )


def mIoU(images_list, masks_list, model):
    res = []
    for image_file, mask_file in zip(images_list, masks_list):
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        mask_tensor = read_image(mask_file, mask=True)
        m = tf.keras.metrics.MeanIoU(num_classes=8)
        m.update_state(prediction_mask, mask_tensor)
        res.append(m.result().numpy())
    return res

"""
### Inference on Test Images
"""

plot_predictions(test_images[:4], colormap, model=history)
print(mIoU(test_images[:4], test_masks[:4], model=history))