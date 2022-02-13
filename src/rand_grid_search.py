"""Permet de réaliser de lancer le programme de machine pour plusieurs hyper paramètres
à la manière du grid search, mais en assurant les mesures des métriques pour affichage
dans le navigateur web via dvc studio, et github, moyennant pour ce dernier l’ajout d’un
fichier de workflow dans le répertoire .github/workflow"""
import subprocess
from random import randint
from random import uniform
from random import seed
from random import choice
from tools import optim_pool

num_exps = 8
data_mix = ['original_version', 'multiplication']

for n in range(num_exps):
    seed(randint(1, 1000))
    rand_nb = uniform
    nb = n % 2
    params = {
        "optimizer": choice(list(optim_pool().keys())),
        "learning_rate": round(rand_nb(0.00001, 0.001), 5),
        "wce_beta": round(rand_nb(0.5, 1), 2),
        "bce_beta": round(rand_nb(0.5, 1), 2)
    }
    subprocess.run(["dvc", "exp", "run", "k2000", "-f", "--queue",
                    "--set-param", f"k2000.data_mix={data_mix[0]}",
                    "--set-param", f"k2000.learning_rate={params['learning_rate']}",
                    "--set-param", f"k2000.optim_type={params['optimizer']}",
                    "--set-param", f"constants.wce_beta={params['wce_beta']}",
                    "--set-param", f"constants.bce_beta={params['wce_beta']}",
                    ])

    subprocess.run(["dvc", "exp", "run", "dolorean", "-f", "--queue",
                    "--set-param", f"dolorean.data_mix={data_mix[0]}",
                    "--set-param", f"dolorean.learning_rate={params['learning_rate']}",
                    "--set-param", f"dolorean.optim_type={params['optimizer']}",
                    "--set-param", f"constants.wce_beta={params['wce_beta']}",
                    "--set-param", f"constants.bce_beta={params['wce_beta']}",
                    ])

    subprocess.run(["dvc", "exp", "run", "k2000", "-f", "--queue",
                    "--set-param", f"k2000.data_mix={data_mix[1]}",
                    "--set-param", f"k2000.learning_rate={params['learning_rate']}",
                    "--set-param", f"k2000.optim_type={params['optimizer']}",
                    "--set-param", f"constants.wce_beta={params['wce_beta']}",
                    "--set-param", f"constants.bce_beta={params['wce_beta']}",
                    ])

    subprocess.run(["dvc", "exp", "run", "dolorean", "-f", "--queue",
                    "--set-param", f"dolorean.data_mix={data_mix[1]}",
                    "--set-param", f"dolorean.learning_rate={params['learning_rate']}",
                    "--set-param", f"dolorean.optim_type={params['optimizer']}",
                    "--set-param", f"constants.wce_beta={params['wce_beta']}",
                    "--set-param", f"constants.bce_beta={params['wce_beta']}",
                    ])
