import pandas as pd
import cv2
import numpy as np
import laerte
from laerte import anticlee

layer_list = anticlee()
dict_label_clr = laerte.sisyphe()


def blur(img_str):
    image = cv2.imread(img_str)
    kernel = 5
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (kernel, kernel), 0)
    return img_blur


def guineapig():
    """Crée tous les masques, en les laissant inscrit sur le disque"""
    for pic_num in range(len(layer_list)):
        if (pic_num % 100) == 0:
            print(f'Nombre de fichiers traités : {str(pic_num)},\nNombre de fichiers restant : {str(len(layer_list) - pic_num)}')
        dfl = pd.read_json(layer_list[pic_num])
        h = dfl.at[0, 'imgHeight']
        w = dfl.at[0, 'imgWidth']

        img_mask = np.zeros(np.hstack((h, w)), dtype='uint8')
        for poly_num in range(len(dfl)):
            label = dfl.at[poly_num, 'objects']['label']
            poly = np.array([dfl.at[poly_num, 'objects']['polygon']])
            img_mask = cv2.fillPoly(img_mask, poly, dict_label_clr[label])
    return


def main():
    guineapig()


if __name__ == "__main__":
    main()
