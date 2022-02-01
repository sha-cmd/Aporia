import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import sys
import seaborn as sns

from scipy.io import loadmat
from tensorflow import keras
from tools import read_image
from glob import glob
from tools import DATA_DIR, IMAGE_SIZE
from dvclive import Live

live = Live()


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims(image_tensor, axis=0))
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


def mIoU(image_file, mask_file, model):
    res = []
    #for image_file, mask_file in zip(images_list, masks_list):
    image_tensor = read_image(image_file)
    prediction_mask = infer(image_tensor=image_tensor, model=model)
    mask_tensor = read_image(mask_file, mask=True)
    m = tf.keras.metrics.MeanIoU(num_classes=8)
    m.update_state(prediction_mask, mask_tensor)
    res.append(m.result().numpy())
    return res


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice = 0
    for index in range(numLabels):
        dice += dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice/numLabels


def main(name="aucun"):
    test_images = sorted(glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/test/**/*.png"), recursive=True))
    test_masks = sorted(glob(os.path.join(DATA_DIR, "finetuning/gtFine/test/**/*octogroups.png"), recursive=True))
    if name == "k2000":
        history = keras.models.load_model('models/k2000')
    elif name == "dolorean":
        history = keras.models.load_model('models/dolorean')
    else:
        sys.exit()
    # Loading the Colormap
    colormap = loadmat(
        "src/city_colormap.mat"
    )["colormap"]
    colormap = colormap * 100
    colormap = colormap.astype(np.uint8)

#    plot_predictions(test_images[:4], colormap, model=history)
    nb = 100
    for i in range(nb):
        if (i % 20) == 0:
            print(f"images traité pour le mIoU : {i}\nimages restantes pour le mIoU {i-nb}\n")
        res = mIoU(test_images[i+1], test_masks[i+1], model=history)
        avg_res = sum(res) / len(res)
        live.log("mIoU", avg_res)
        live.next_step()
    df = pd.read_csv("dvclive/mIoU.tsv", sep='\t')
    df = df.rename(columns={'step': 'pics'})
    df.to_csv(name + "/mIoU.csv", sep=',', index_label='index')
    data = df['mIoU'].apply(lambda x: round(x, 5))
    sns.distplot(data, hist=False)
    plt.xlabel('mIoU')
    plt.ylabel('Density')
    plt.title('Density mIoU ' + name)
    plt.tight_layout()
    plt.savefig(name + '/mIoU_density.jpg')


    # Dice coefficient section
    # need a serious code mutation
    # to correspond to the needs of my function
    # num_class = 8
    # image_tensor = read_image(test_images[1])
    # imgA = infer(image_tensor=image_tensor, model=history)
    # imgB = Image.open(test_masks[1]).resize((IMAGE_SIZE, IMAGE_SIZE))  # np.random.randint(low=0, high= 2, size=(5, 64, 64, num_class) )
    #
    # plt.imshow(imgA)  # for 0th image, class 0 map
    # plt.show()
    #
    # plt.imshow(imgB)  # for 0th image, class 0 map
    # plt.show()
    #
    # dice_score = dice_coef_multilabel(imgA, imgB, num_class)
    # print(f'For A and B {dice_score}')
    #
    # dice_score = dice_coef_multilabel(imgA, imgA, num_class)
    # print(f'For A and A {dice_score}')


if __name__ == "__main__":
    main()
