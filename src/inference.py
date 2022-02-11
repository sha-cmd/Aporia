import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import seaborn as sns
import tensorflow as tf
import yaml

from objects.TimeCallBack import TimingCallback
from dvclive import Live
from glob import glob
from objects.WeightedCrossEntropy import WeightedCrossEntropy
from objects.BalancedCrossEntropy import BalancedCrossEntropy
from scipy.io import loadmat
from tensorflow import keras
from tools import read_image
from tools import DATA_DIR, IMAGE_SIZE
from objects.TimeCallBack import TimingCallback

live = Live()

global name


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
    global name
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.savefig(name + '/inference.jpg')


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
        dice += dice_coef(np.where(y_true == index, 1, 0), np.where(y_pred == index, 1, 0))
    return dice / numLabels


def plot(model="aucun", name="aucun"):
    df = pd.read_csv("dvclive/" + name + ".tsv", sep='\t')
    df = df.rename(columns={'step': 'pics'})
    df.to_csv(model + "/" + name + ".csv", sep=',', index_label='index')
    data = df[name].apply(lambda x: round(x, 5))
    sns.displot(data, kind='kde')
    plt.xlabel(name)
    plt.ylabel('Density')
    plt.title('Density ' + name + " " + model)
    plt.tight_layout()
    plt.savefig(model + '/' + name + '_density.jpg')


def main(test_images, test_masks, model="aucun"):
    global name
    name = model
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        test_size = int(params[model]['test_size'])
    print(model)
    #test_images = sorted(glob(os.path.join(DATA_DIR, "coarse_tuning/leftImg8bit/train/**/*.png"), recursive=True))[
    #              -test_size:]
    #test_masks = sorted(glob(os.path.join(DATA_DIR, "finetuning/gtFine/train/**/*octogroups.png"), recursive=True))[
    #             -test_size:]
    metrics_wce = WeightedCrossEntropy
    metrics_bce = BalancedCrossEntropy
    cb = TimingCallback()

    history = keras.models.load_model('models/' + model, compile=False,
                                      custom_objects={'WeightedCrossEntropy': metrics_wce,
                                                      'BalancedCrossEntropy': metrics_bce,
                                                      'TimingCallback': cb})

    # Loading the Colormap
    colormap = loadmat(
        "src/city_colormap.mat"
    )["colormap"]
    colormap = colormap * 100
    colormap = colormap.astype(np.uint8)
    plot_predictions(test_images[4:5], colormap, model=history)
    nb = test_size
    for i in range(nb):
        if (i % 20) == 0:
            print(f"images traité pour le mIoU : {i}\nimages restantes pour le mIoU {i - nb}\n")
        res = mIoU(test_images[i], test_masks[i], model=history)
        avg_res = sum(res) / len(res)
        live.log("mIoU", round(avg_res, 2))
        live.next_step()
    plot(model, 'mIoU')

    num_class = 8
    for i in range(nb):
        if (i % 20) == 0:
            print(f"images traité pour le Dice : {i}\nimages restantes pour le Dice {i - nb}\n")
        image_tensor = read_image(test_images[i])
        imgA = infer(image_tensor=image_tensor, model=history)
        imgB = read_image(test_masks[i], mask=True)
        imgB = np.array(imgB.numpy().reshape(256, 256), dtype='int')
        dice_score = dice_coef_multilabel(imgA, imgB, num_class)
        live.log("Dice_coefficient", round(dice_score, 2))
        live.next_step()
    plot(model, 'Dice_coefficient')


if __name__ == "__main__":
    main()
