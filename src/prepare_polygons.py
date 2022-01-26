import pandas as pd
from laerte import ctimene
from laerte import ulysse

global mask_list
global dict_label
mask_list = ulysse()
dict_label = ctimene()


def marlboro():
    global mask_list
    global dict_label
    # Boucle pour transformer les fichiers en 8 groupes
    for i in range(len(mask_list)):
        if (i % 100) == 0:
            print(f'Nombre de fichiers traités : {str(i)},\nNombre de fichiers restant : {str(len(mask_list) - i)}')
        num_ind = pd.read_json(mask_list[i])['objects'].shape[0]
        df = pd.read_json(mask_list[i])
        for j in range(num_ind):
            old_lab = df.at[j, 'objects']['label']
            # Reduce number of category
            df.at[j, 'objects']['label'] = dict_label[old_lab]
        df.to_json(mask_list[i][:-5] + '_octogroups.json', orient='columns')
        del df


def main():
    marlboro()


if __name__ == "__main__":
    main()