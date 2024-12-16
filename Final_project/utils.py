import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from Symbol import SYMBOLS
import os
import matplotlib as cm

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
    elif type == "catsanddogs":
        train_path = os.path.join(os.getcwd(), "datasets/classification/CatsDogs/CatsDogs_TRAIN.ts")
        test_path = os.path.join(os.getcwd(), "datasets/classification/CatsDogs/CatsDogs_TEST.ts")
    elif type == "acsf1":
        train_path = os.path.join(os.getcwd(), "datasets/classification/ACSF1/ACSF1_TRAIN.ts")
        test_path = os.path.join(os.getcwd(), "datasets/classification/ACSF1/ACSF1_TEST.ts")

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

def describe_ecg_dataset(x_train, y_train, x_test, y_test):

    print(f"Number of examples in the training set: {x_train.shape[0]}")
    print(f"Number of examples in the test set: {x_test.shape[0]}")
    print(f"Length of the time series: {x_train.shape[1]}")
    print("\n")

    # Class distribution
    print("Class distribution (Training set):")
    print(y_train.value_counts().sort_index())
    print("\nClass distribution (Test set):")
    print(y_test.value_counts().sort_index())
    print("\n")

def plot_classes(df0, df1, num_seg):

    # Convert to numeric (force errors to NaN) for df1
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    df1 = df1.fillna(0)  # Fill NaN with 0

    # Calculation for df1
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

    # Convert to numeric (force errors to NaN) for df0
    df0 = df0.apply(pd.to_numeric, errors='coerce')
    df0 = df0.fillna(0)  # Fill NaN with 0

    # Calculation for df0
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

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # First subplot - df1
    sns.lineplot(ax=axes[0], x=mean_series1.index, y=mean_series1, label="Mean", color="blue")
    axes[0].fill_between(mean_series1.index, ci_low1, ci_high1, color="blue", alpha=0.2, label="95% CI")
    if num_seg is not None:
        for i in range(num_seg+1):
            axes[0].axvline(x= i * (len(mean_series1.index) // num_seg), color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("Class 1", fontsize=14)
    axes[0].set_xlabel("Index", fontsize=12)
    axes[0].set_ylabel("Value", fontsize=12)
    axes[0].legend()

    axes[0].grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Second subplot - df0
    sns.lineplot(ax=axes[1], x=mean_series0.index, y=mean_series0, label="Mean", color="blue")
    axes[1].fill_between(mean_series0.index, ci_low0, ci_high0, color="blue", alpha=0.2, label="95% CI")
    if num_seg is not None:
        for i in range(num_seg+1):
            axes[1].axvline(x= i * (len(mean_series0.index) // num_seg), color="red", linestyle="--", alpha=0.7)
    axes[1].set_title("Class 2", fontsize=14)
    axes[1].set_xlabel("Index", fontsize=12)
    axes[1].set_ylabel("Value", fontsize=12)
    axes[1].legend()

    axes[1].grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)


    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

def plot_multiple_classes(df_list, num_seg):

    mean_df = []
    low_df = []
    high_df = []

    for df1 in df_list:
        # Convert to numeric (force errors to NaN) for df1
        df1 = df1.apply(pd.to_numeric, errors='coerce')
        df1 = df1.fillna(0)  # Fill NaN with 0

        # Calculation for df1
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

        mean_df.append(mean_series1)
        low_df.append(ci_low1)
        high_df.append(ci_high1)

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(5, 2, figsize=(16, 12))

    # Flatten the axes for easier iteration (instead of a 2D matrix)
    axes_flat = axes.flat

    for i in range(len(df_list)):
        ax = axes_flat[i]
        sns.lineplot(ax=ax, x=mean_df[i].index, y=mean_df[i], label="Mean", color="blue")
        ax.fill_between(mean_df[i].index, low_df[i], high_df[i], color="blue", alpha=0.2, label="95% CI")
        if num_seg is not None:
            for i in range(num_seg+1):
                ax.axvline(x= i * (len(mean_df[i].index) // num_seg), color="red", linestyle="--", alpha=0.7)
        ax.set_title("Class 1", fontsize=14)
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()

        ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

def plot_acf_mean_series(df_list, data_type = "2class", conf_level=0.95):
    """
    Plots the autocorrelation function of the mean time series for each dataset in df_list,
    with a 95% confidence interval band.
    
    Parameters:
    - df_list : list of DataFrames. Each DataFrame contains n time series of length p.
    - conf_level : float, confidence level for the confidence interval (default 95%).
    """
    if data_type == "2class":
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Flatten the axes for easier iteration (instead of a 2D matrix)
        axes_flat = axes.flat

    else: 
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(5, 2, figsize=(16, 12))

        # Flatten the axes for easier iteration (instead of a 2D matrix)
        axes_flat = axes.flat

    for idx, df in enumerate(df_list):
        ax = axes_flat[idx]
        print(f"Processing class {idx + 1}...")

        # Compute the mean time series
        mean_series = df.mean(axis=0)

        # Compute autocorrelation and confidence intervals
        acf_values, conf_int = acf(mean_series, alpha=1 - conf_level, fft=True, nlags=len(mean_series)-1)
        
        # Confidence interval
        lower_bound, upper_bound = conf_int[:, 0], conf_int[:, 1]
        
        # Plot the ACF with confidence intervals
        ax.plot(acf_values, label='Autocorrelation', color='blue')
        ax.fill_between(range(len(acf_values)), lower_bound, upper_bound, color='lightblue', alpha=0.5, label=f'{int(conf_level*100)}% Confidence Interval')
        ax.axhline(0, linestyle='--', color='gray', linewidth=1)

        # Add labels, title, and legend
        ax.set_title(f"Autocorrelation Function (Class {idx + 1})", fontsize=14)
        ax.set_xlabel("Lag", fontsize=12)
        ax.set_ylabel("Autocorrelation", fontsize=12)
        ax.legend(fontsize=10)

        ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)


    plt.tight_layout()
    plt.show()

def plot_periodogram_mean_series(df_list, data_type = "2class", sampling_frequency=1.0):
    """
    Plots the periodogram of the mean time series for each dataset in df_list.
    
    Parameters:
    - df_list : list of DataFrames. Each DataFrame contains n time series of length p.
    - sampling_frequency : float, the sampling frequency of the time series (default is 1.0).
    """
    if data_type == "2class":
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Flatten the axes for easier iteration (instead of a 2D matrix)
        axes_flat = axes.flat

    else: 
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(5, 2, figsize=(16, 12))

        # Flatten the axes for easier iteration (instead of a 2D matrix)
        axes_flat = axes.flat

    for idx, df in enumerate(df_list):
        ax = axes_flat[idx]

        print(f"Processing dataset {idx + 1}...")
        
        # Compute the mean time series
        mean_series = df.mean(axis=0)
        
        # Compute the periodogram
        frequencies, power_spectrum = signal.periodogram(mean_series, sampling_frequency)
        
        # Plot the periodogram
        ax.plot(frequencies, power_spectrum, color='blue', lw=1.5, label='Power Spectrum')
        
        # Highlight the dominant frequency
        dominant_frequency_idx = np.argmax(power_spectrum)
        dominant_frequency = frequencies[dominant_frequency_idx]
        ax.axvline(dominant_frequency, color='red', linestyle='--', 
                   label=f'Dominant Frequency = {dominant_frequency:.3f} Hz')

        # Add labels, title, and legend
        ax.set_title(f"Periodogram (Class {idx + 1})", fontsize=14)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Power Spectral Density", fontsize=12)
        ax.legend(fontsize=10)

        ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    plt.tight_layout()
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



