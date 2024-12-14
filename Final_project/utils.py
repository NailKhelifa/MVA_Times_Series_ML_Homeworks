import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################################################################
###################################################### DATA LOADING ######################################################
##########################################################################################################################
def load_control_chart_dataset(filepath):
    """
    Charge un dataset ASCII de séries temporelles, ajoute les labels et des noms explicites pour chaque série.
    
    Args:
        filepath (str): Chemin vers le fichier ASCII contenant le dataset.
        
    Returns:
        pd.DataFrame: Dataset avec les colonnes des valeurs, des noms de séries et des labels.
    """
    # Charger les données sans en-tête
    data = pd.read_csv(filepath, header=None, delim_whitespace=True)
    
    # Ajouter une colonne de labels en fonction des classes définies
    labels = []
    for i in range(len(data)):
        if 0 <= i < 100:
            labels.append("Normal")
        elif 100 <= i < 200:
            labels.append("Cyclic")
        elif 200 <= i < 300:
            labels.append("Increasing trend")
        elif 300 <= i < 400:
            labels.append("Decreasing trend")
        elif 400 <= i < 500:
            labels.append("Upward shift")
        elif 500 <= i < 600:
            labels.append("Downward shift")
    
    data["Label"] = labels
    
    return data

def load_CBF_dataset(path_test, path_train):
    """
    Charge les jeux de données d'entraînement et de test à partir de fichiers CSV délimités par des espaces.
    
    Arguments:
    - path_train (str): Chemin vers le fichier de données d'entraînement.
    - path_test (str): Chemin vers le fichier de données de test.
    
    Retourne:
    - cbf_df_train (DataFrame): DataFrame pour le jeu d'entraînement avec des labels entiers.
    - cbf_df_test (DataFrame): DataFrame pour le jeu de test avec des labels entiers.
    """
    
    # Chargement du fichier d'entraînement
    cbf_df_train = pd.read_csv(path_train, header=None, delim_whitespace=True)
    cbf_df_train.index = cbf_df_train.index.astype(int)  # Conversion de l'index en int
    cbf_df_train.rename(columns={0: 'labels'}, inplace=True)  # Renommage de la colonne 0 en 'labels'
    cbf_df_train['labels'] = cbf_df_train['labels'].astype(int)  # Conversion des labels en int

    # Chargement du fichier de test
    cbf_df_test = pd.read_csv(path_test, header=None, delim_whitespace=True)
    cbf_df_test.index = cbf_df_test.index.astype(int)  # Conversion de l'index en int
    cbf_df_test.rename(columns={0: 'labels'}, inplace=True)  # Renommage de la colonne 0 en 'labels'
    cbf_df_test['labels'] = cbf_df_test['labels'].astype(int)  # Conversion des labels en int  

    # Retour des DataFrames d'entraînement et de test
    return cbf_df_train, cbf_df_test

##########################################################################################################################
####################################################### UTILS FUNC #######################################################
##########################################################################################################################

def plot_classes(df0, df1, num_seg):

    # Convertir en numérique (forcer les erreurs à NaN) pour df1
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    df1 = df1.fillna(0)  # Remplir les NaN avec 0

    # Calcul pour df1
    mean_series1 = df1.mean(axis=0)
    ci_low1 = mean_series1 - 1.96 * df1.sem(axis=0)
    ci_high1 = mean_series1 + 1.96 * df1.sem(axis=0)

    mean_series1.index = pd.to_numeric(mean_series1.index, errors='coerce')
    ci_low1 = pd.to_numeric(ci_low1, errors='coerce')
    ci_high1 = pd.to_numeric(ci_high1, errors='coerce')

    valid_indices = ~ci_low1.isna() & ~ci_high1.isna()
    mean_series1 = mean_series1[valid_indices]
    ci_low1 = ci_low1[valid_indices]
    ci_high1 = ci_high1[valid_indices]

    # Convertir en numérique (forcer les erreurs à NaN) pour df0
    df0 = df0.apply(pd.to_numeric, errors='coerce')
    df0 = df0.fillna(0)  # Remplir les NaN avec 0

    # Calcul pour df0
    mean_series0 = df0.mean(axis=0)
    ci_low0 = mean_series0 - 1.96 * df0.sem(axis=0)
    ci_high0 = mean_series0 + 1.96 * df0.sem(axis=0)

    mean_series0.index = pd.to_numeric(mean_series0.index, errors='coerce')
    ci_low0 = pd.to_numeric(ci_low0, errors='coerce')
    ci_high0 = pd.to_numeric(ci_high0, errors='coerce')

    valid_indices0 = ~ci_low0.isna() & ~ci_high0.isna()
    mean_series0 = mean_series0[valid_indices0]
    ci_low0 = ci_low0[valid_indices0]
    ci_high0 = ci_high0[valid_indices0]

    # Créer une figure avec deux sous-graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Premier subplot - df1
    sns.lineplot(ax=axes[0], x=mean_series1.index, y=mean_series1, label="Moyenne", color="blue")
    axes[0].fill_between(mean_series1.index, ci_low1, ci_high1, color="blue", alpha=0.2, label="IC 95%")
    if num_seg is not None:
        for i in range(num_seg+1):
            axes[0].axvline(x= i * (len(mean_series1.index) // num_seg), color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("Ischemia heartbeat - moyenne avec IC 95%", fontsize=14)
    axes[0].set_xlabel("Index", fontsize=12)
    axes[0].set_ylabel("Valeur", fontsize=12)
    axes[0].legend()

    # Deuxième subplot - df0
    sns.lineplot(ax=axes[1], x=mean_series0.index, y=mean_series0, label="Moyenne", color="blue")
    axes[1].fill_between(mean_series0.index, ci_low0, ci_high0, color="blue", alpha=0.2, label="IC 95%")
    if num_seg is not None:
        for i in range(num_seg+1):
            axes[1].axvline(x= i * (len(mean_series0.index) // num_seg), color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Normal heartbeat - moyenne avec IC 95%", fontsize=14)
    axes[1].set_xlabel("Index", fontsize=12)
    axes[1].set_ylabel("Valeur", fontsize=12)
    axes[1].legend()

    # Ajuster l'espacement entre les subplots
    plt.tight_layout()
    plt.show()





