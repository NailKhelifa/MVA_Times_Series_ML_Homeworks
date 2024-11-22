import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd

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


def sax_transform(time_series, alphabet_size=5, segments=4):
    """
    Transforms a time series into its SAX representation.

    Args:
        time_series (list or np.ndarray): The input time series data.
        alphabet_size (int): The number of discrete symbols in the SAX alphabet.
        segments (int): The number of segments for Piecewise Aggregate Approximation (PAA).

    Returns:
        str: SAX representation of the time series.
    """
    # Step 1: Normalize the time series
    normalized_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    # Step 2: Perform PAA (Piecewise Aggregate Approximation)
    def paa_transform(series, segments):
        n = len(series)
        segment_size = n // segments
        paa = []
        for i in range(0, n, segment_size):
            paa.append(np.mean(series[i:i+segment_size]))
        return np.array(paa)
    
    paa_representation = paa_transform(normalized_series, segments)
    
    # Step 3: Discretize into SAX symbols
    def get_breakpoints(alphabet_size):
        return stats.norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    
    breakpoints = get_breakpoints(alphabet_size)
    alphabet = "abcdefghijklmnopqrstuvwxyz"[:alphabet_size]
    
    sax_symbols = []
    for value in paa_representation:
        for i, bp in enumerate(breakpoints):
            if value < bp:
                sax_symbols.append(alphabet[i])
                break
        else:
            sax_symbols.append(alphabet[-1])
    
    return ''.join(sax_symbols)
