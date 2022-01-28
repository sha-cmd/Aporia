import os
import glob
import pandas as pd


def main():
    dirs = glob.glob('k2000')
    dirs_dict = {}
    for dir in dirs:
        metrics_dict = {}
        files = glob.glob(dir + '/*tsv')
        for file in files:
            train_val_dict = {}
            header = file.split('/')[1][:4]
            if not (header == 'val_'):
                metric = [x.split('.') for x in file.split('/')][1][0]
                metric_file = file.split('/')[1]
                train_val_dict.update({'train': dir + '/' + metric_file,
                                       'val': dir + '/' + 'val_' + metric_file})

                metrics_dict.update({metric: train_val_dict})
        # Création d’une entrée pour chaque répertoire d’intérêt
        dirs_dict.update({dir: metrics_dict})
    try:
        os.remove('yamlhadoc.txt')
    except FileNotFoundError as e:
        print(e)

    # Itération à l’intérieur de notre dictionnaire de dictionnaire
    for key_dir, metrics in dirs_dict.items():
        for key_metric, learning in metrics.items():
            # Nous réalisons des actions de transformation
            # pour conformer nos données avec le modèle .dvc/plots/multi-plots.json
            df_train = pd.read_csv(learning['train'], sep='\t')
            df_train['stage'] = 'train'
            df_val = pd.read_csv(learning['val'], sep='\t')
            df_val['stage'] = 'val'
            df_list = [df_train, df_val]
            for df in df_list:
                new_cols = []
                for col in df.columns:
                    if 'val_' in col:
                        new_cols.append(col[4:])
                    elif 'step' in col:
                        new_cols.append('epoch')
                    else:
                        new_cols.append(col)
                df.columns = pd.Index(new_cols)
            df = pd.concat([df_train, df_val])
            df.to_csv(key_dir + '/' + key_metric + '_multi_plots.csv', index_label='index')
            # Fichier pour rajouter les plots dans dvc.yaml
            with open('yamlhadoc.txt', 'a') as f:
                f.writelines(f"- {key_dir}/{key_metric}_multi_plots.csv:\n\
                cache: false\n\
                persist: true\n\
                title: Train/Test {' '.join(key_metric.split('_')).title()} {key_dir}\n\
                template: multi_loss\n\
                x: epoch\n\
                y: {key_metric}\n\
                \n")


if __name__ == "__main__":
    main()
