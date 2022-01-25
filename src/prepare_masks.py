import glob
import pandas as pd
import cv2
import numpy as np

dict_label = {'road': 'flat', 'sidewalk': 'flat', 'parking': 'flat', 'rail track': 'flat',
              'person': 'human', 'rider': 'human',
              'truck': 'vehicle', 'car': 'vehicle', 'bus': 'vehicle', 'on rails': 'vehicle',
              'motorcycle': 'vehicle', 'bicycle': 'vehicle', 'caravan': 'vehicle',
              'trailer': 'vehicle',
              'building': 'construction', 'wall': 'construction', 'fence': 'construction', 'guard rail': 'construction',
              'bridge': 'construction', 'tunnel': 'construction',
              'pole': 'object', 'pole group': 'object', 'traffic sign': 'object',
              'traffic light': 'object',
              'vegetation': 'nature', 'terrain': 'nature',
              'ground': 'void', 'dynamic': 'void', 'static': 'void',
              'flat': 'flat', 'human': 'human', 'vehicle': 'vehicle', 'construction': 'construction',
              'object': 'object', 'nature': 'nature', 'void': 'void', 'sky': 'sky',
              'license plate': 'vehicle',
              'ego vehicle': 'void',
              'out of roi': 'void',
              'bicyclegroup': 'vehicle',
              'cargroup': 'vehicle',
              'persongroup': 'human',
              'polegroup': 'object',
              'rectification border': 'void',
              'train': 'vehicle',
              'ridergroup': 'vehicle',
              'motorcyclegroup': 'vehicle',
              'truckgroup': 'vehicle'
              }

global mask_list
mask_list = glob.glob('../data/finetuning/gtFine/**/*polygons.json', recursive=True)


def blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    return blur


def marlboro():
    global mask_list
    # Boucle pour transformer les fichiers en 8 groupes
    for i in range(len(mask_list)):
        num_ind = pd.read_json(mask_list[i])['objects'].shape[0]
        df = pd.read_json(mask_list[i])
        for j in range(num_ind):
            old_lab = df.at[j, 'objects']['label']
            # Reduce number of category
            df.at[j, 'objects']['label'] = dict_label[old_lab]
        df.to_json(mask_list[i][:-5] + '_octogroups.json', orient='columns')
        del df


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
        img_name = layer_list[pic_num][:7] + '/coarse_tuning/leftImg8bit/' \
                        + layer_list[pic_num].split('/')[-3] + '/' \
                        + layer_list[pic_num].split('/')[-2] + '/' \
                        + layer_list[pic_num].split('/')[-1][:-31] + 'leftImg8bit.png'
        image = cv2.imread(img_name)
        masked_image = cv2.bitwise_and(blur(image), img_mask)
        img_mask_name = layer_list[pic_num][:-4] + 'png'
        cv2.imwrite(img_mask_name, img_mask)
    return

marlboro()
guineapig()