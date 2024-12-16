import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from Symbol import SYMBOLS
import os

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

def generate_data(type="ECG200"):
    if type == "ECG200":
        train_path = os.path.join(os.getcwd(), "datasets/classification/ECG200/ECG200_TRAIN.ts")
        test_path = os.path.join(os.getcwd(), "datasets/classification/ECG200/ECG200_TEST.ts")
    elif type == "computers":
        train_path = os.path.join(os.getcwd(), "datasets/classification/Computers/Computers_TRAIN.ts")
        test_path = os.path.join(os.getcwd(), "datasets/classification/Computers/Computers_TEST.ts")
    elif type == "adiac":
        train_path = os.path.join(os.getcwd(), "datasets/classification/Adiac/Adiac_TRAIN.ts")
        test_path = os.path.join(os.getcwd(), "datasets/classification/Adiac/Adiac_TEST.ts")

    X_train = pd.read_csv(train_path, 
                        sep=",", 
                        header=None
                        )

    X_train.columns = list(X_train.columns[:-1]) + ['label']
    x_train, y_train = X_train.iloc[:, :-1], X_train["label"]

    X_test = pd.read_csv(test_path, 
                      sep=",", 
                      header=None
                      )

    X_test.columns = list(X_test.columns[:-1]) + ['label']
    x_test, y_test = X_test.iloc[:, :-1], X_test["label"]

    return X_train, x_train, y_train, X_test, x_test, y_test

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
    axes[0].set_title("Classe 1", fontsize=14)
    axes[0].set_xlabel("Index", fontsize=12)
    axes[0].set_ylabel("Valeur", fontsize=12)
    axes[0].legend()

    # Deuxième subplot - df0
    sns.lineplot(ax=axes[1], x=mean_series0.index, y=mean_series0, label="Moyenne", color="blue")
    axes[1].fill_between(mean_series0.index, ci_low0, ci_high0, color="blue", alpha=0.2, label="IC 95%")
    if num_seg is not None:
        for i in range(num_seg+1):
            axes[1].axvline(x= i * (len(mean_series0.index) // num_seg), color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Classe 2", fontsize=14)
    axes[1].set_xlabel("Index", fontsize=12)
    axes[1].set_ylabel("Valeur", fontsize=12)
    axes[1].legend()

    # Ajuster l'espacement entre les subplots
    plt.tight_layout()
    plt.show()


def plot_classes_multisubplots(data, num_classes, num_seg=None):
    """
    Plot multiple classes in subplots with confidence intervals.
    
    Parameters:
    - data: DataFrame contenant les données avec une colonne "label" indiquant les classes.
    - num_classes: Nombre total de classes dans les données.
    - num_seg: Nombre de segments pour ajouter des lignes verticales (optionnel).
    """
    # Vérifier que les colonnes autres que "label" contiennent des données numériques
    numeric_columns = data.columns.difference(['label'])
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Remplacer les NaN par 0 (ou appliquer une autre stratégie si nécessaire)
    data[numeric_columns] = data[numeric_columns].fillna(0)

    # Créer une grille de subplots
    num_rows = (num_classes // 6) + (1 if num_classes % 6 != 0 else 0)  # 6 colonnes par ligne
    num_cols = 6  # Nombre maximum de colonnes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4), constrained_layout=True)
    axes = axes.ravel()  # Aplatir les subplots pour une gestion plus simple

    # Itérer sur chaque classe
    for i in range(num_classes):
        ax = axes[i]
        # Filtrer les données pour la classe actuelle
        df_class = data[data["label"] == i + 1].iloc[:, :-1]  # Exclure la colonne "label"
        
        # Convertir en numérique et gérer les NaN (déjà fait, mais peut être rappelé ici si nécessaire)
        df_class = df_class.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Calculer la moyenne et les intervalles de confiance
        mean_series = df_class.mean(axis=0)
        ci_low = mean_series - 1.96 * df_class.sem(axis=0)
        ci_high = mean_series + 1.96 * df_class.sem(axis=0)

        # Traçage de la moyenne et des intervalles de confiance
        sns.lineplot(ax=ax, x=mean_series.index, y=mean_series, label=f"Classe {i + 1}", color="blue")
        ax.fill_between(mean_series.index, ci_low, ci_high, color="blue", alpha=0.2, label="IC 95%")
        
        # Ajouter des segments verticaux si `num_seg` est défini
        if num_seg is not None:
            for j in range(num_seg + 1):
                ax.axvline(x=j * (len(mean_series.index) // num_seg), color="red", linestyle="--", alpha=0.7)
        
        # Ajouter des titres et des légendes
        ax.set_title(f"Classe {i + 1}", fontsize=12)
        ax.set_xlabel("Index", fontsize=10)
        ax.set_ylabel("Valeur", fontsize=10)
        ax.legend(fontsize=8)

    # Supprimer les subplots inutilisés si num_classes < num_rows * num_cols
    for j in range(num_classes, len(axes)):
        fig.delaxes(axes[j])

    # Ajouter un titre global
    fig.suptitle("Visualisation des classes", fontsize=16)
    plt.show()


##########################################################################################################################
################################ BEFORE SYMOBLIZATION:  PRE-PROCESSING SERIES ############################################
##########################################################################################################################

def std_scaler(df):
    """
    Standardise les séries temporelles dans un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les séries temporelles.
        
    Returns:
        pd.DataFrame: DataFrame avec les séries temporelles standardisées.
    """
    means = df.mean(axis=0)
    df_centered = df - means

    std_devs = df.std(axis=0)
    df_normalized = df_centered / std_devs  

    return df_normalized

def describe_ecg_dataset(x_train, y_train, x_test, y_test):

    print(f"Nombre d'exemples dans l'ensemble d'entraînement : {x_train.shape[0]}")
    print(f"Nombre d'exemples dans l'ensemble de test : {x_test.shape[0]}")
    print(f"Longueur des séries temporelles : {x_train.shape[1]}")
    print("\n")

    # Répartition des classes
    print("Répartition des classes (Ensemble d'entraînement) :")
    print(y_train.value_counts().sort_index())
    print("\nRépartition des classes (Ensemble de test) :")
    print(y_test.value_counts().sort_index())
    print("\n")

def plot_KNN_accuracies(X_train, X_test):
    
    abs_x = np.arange(2, 15)
    sax_acc = np.zeros(13)
    esax_acc = np.zeros(13)
    tsax_acc = np.zeros(13)
    oneD_sax_acc = np.zeros(13)
    sfa_acc = np.zeros(13)
    boss_acc = np.zeros(13)
    astride_acc = np.zeros(13)

    for i in range(2, 15):

        SAX_KNN = SYMBOLS(
                    X_train, 
                    X_test, 
                    'SAX', 
                    num_segments=10, 
                    alphabet_size=i)
        SAX_KNN._predict()
        sax_acc[i-2] = SAX_KNN.accuracy

        ESAX_KNN = SYMBOLS(
                    X_train, 
                    X_test, 
                    'ESAX', 
                    num_segments=10, 
                    alphabet_size=i)
        ESAX_KNN._predict()
        esax_acc[i-2] = ESAX_KNN.accuracy

        #TSAX_KNN = SYMBOLS(
                    #X_train, 
                    #X_test, 
                    #'TSAX', 
                    #num_segments=10, 
                    #alphabet_size=i)
        #TSAX_KNN._predict()
        #tsax_acc[i-2] = TSAX_KNN.accuracy

        SFA_KNN = SYMBOLS(
                    X_train, 
                    X_test, 
                    'SFA',
                    num_segments=10,
                    alphabet_size=i)
        SFA_KNN.predict_SFA()
        sfa_acc[i-2] = SFA_KNN.accuracy

        BOSS_KNN = SYMBOLS(
                    X_train, 
                    X_test, 
                    'BOSS', 
                    num_segments=10, 
                    alphabet_size=i, word_size=2, window_size=10)
        BOSS_KNN.predict_BOSS()
        boss_acc[i-2] = BOSS_KNN.accuracy

        ASTRIDE_KNN = SYMBOLS(
                    X_train, 
                    X_test, 
                    'ASTRIDE', 
                    num_segments=10, 
                    alphabet_size=i)
        ASTRIDE_KNN.predict_ASTRIDE()
        astride_acc[i-2] = ASTRIDE_KNN.accuracy

    np.save('sax_acc.npy', sax_acc)
    np.save('esax_acc.npy', esax_acc)
    np.save('tsax_acc.npy', tsax_acc)
    np.save('oneD_sax_acc.npy', oneD_sax_acc)
    np.save('sfa_acc.npy', sfa_acc)
    np.save('boss_acc.npy', boss_acc)
    np.save('astride_acc.npy', astride_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(abs_x, sax_acc, color='blue', label="SAX")
    plt.plot(abs_x, esax_acc, color='red', label="ESAX")
    plt.plot(abs_x, tsax_acc, color='green', label="TSAX")
    plt.plot(abs_x, oneD_sax_acc, color='purple', label="1D SAX")
    plt.plot(abs_x, sfa_acc, color='orange', label="SFA")
    plt.plot(abs_x, boss_acc, color='black', label="BOSS")
    plt.plot(abs_x, astride_acc, color='brown', label="ASTRIDE")
    plt.title("Accuracy en fonction de la taille de l'alphabet - 10 segments", fontsize=14)
    plt.xlabel("Taille de l'alphabet", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.show()


def plot_3D(x_axis, y_axis, z_axis, title, x_label, y_label, z_label): 
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = z_axis

    # On convertit en données plates
    x_flat = X.ravel()  
    y_flat = Y.ravel()  
    z_flat = np.zeros_like(x_flat) 
    dz = Z.ravel()  

    dx = dy = 0.8  # Largeur des barres 

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = cm.coolwarm(dz / dz.max())  # Normaliser les hauteurs (dz) pour utiliser la colormap

    ax.bar3d(x_flat, y_flat, z_flat, dx, dy, dz, shade=True, color=colors, edgecolor='black', alpha=0.9)

    ax.invert_yaxis() 
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.set_zlabel(z_label, labelpad=10)
    plt.title(title, fontsize=16)

    ax.view_init(elev=30, azim=120) 
    plt.show()

def plot_recons(TS_test, r1, r2, r3, r4, method): 
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(TS_test), color='blue', label="Série Initiale")
    plt.plot(r1, color='red', label=f"Reconstruction from {method} - 10 segments, 4 symboles")
    plt.plot(r2, color='orange', label=f"Reconstruction from {method} - 20 segments, 4 symboles")
    plt.plot(r3, color='green', label=f"Reconstruction from {method} - 10 segments, 8 symboles")
    plt.plot(r4, color='black', label=f"Reconstruction from {method} - 20 segments, 8 symboles")
    plt.title(f"Différentes reconstructions à partir de {method}", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Valeurs", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.show()



