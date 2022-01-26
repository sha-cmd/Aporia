import glob
import pandas as pd
import cv2
import numpy as np
from laerte import ctimene

dict_label = ctimene()


def blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur_charm = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return blur_charm


def guineapig():
    """Crée tous les masques, en les laissant inscrit sur le disque"""
    layer_list = sorted(glob.glob('../data/finetuning/gtFine/**/*octogroups.json', recursive=True))
    for pic_num in range(len(layer_list)):
        dfl = pd.read_json(layer_list[pic_num])
        h = dfl.at[0, 'imgHeight']
        w = dfl.at[0, 'imgWidth']
        dict_label_clr = {'construction': 60,
                          'flat': 30,
                          'human': 100,
                          'nature': 80,
                          'object': 20,
                          'sky': 40,
                          'vehicle': 120,
                          'void': 10}

        img_mask = np.zeros(np.hstack((h, w)), dtype='uint8')
        for poly_num in range(len(dfl)):
            label = dfl.at[poly_num, 'objects']['label']
            poly = np.array([dfl.at[poly_num, 'objects']['polygon']])
            img_mask = cv2.fillPoly(img_mask, poly, dict_label_clr[label])
        # Stock of those lines in comment mode, in case of inference necessities
        #img_name = layer_list[pic_num][:7] + '/coarse_tuning/leftImg8bit/' \
        #           + layer_list[pic_num].split('/')[-3] + '/' \
        #           + layer_list[pic_num].split('/')[-2] + '/' \
        #           + layer_list[pic_num].split('/')[-1][:-31] + 'leftImg8bit.png'
        # image = cv2.imread(img_name)
        # masked_image = cv2.bitwise_and(blur(image), img_mask)
        img_mask_name = layer_list[pic_num][:-4] + 'png'
        cv2.imwrite(img_mask_name, img_mask)
    return


guineapig()
