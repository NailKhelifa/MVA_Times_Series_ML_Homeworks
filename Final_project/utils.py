import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm

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

def compute_linear_regression(t, v):
    """
    Computes the linear regression parameters (slope and intercept) for a given time segment.
    t: Time values
    v: Time series values
    Returns:
        - s: slope
        - a: average value of the linear regression over the segment
    """
    T_mean = np.mean(t)
    V_mean = np.mean(v)

    numerator = np.sum((t - T_mean) * (v - V_mean))
    denominator = np.sum((t - T_mean) ** 2)

    s = numerator / denominator
    b = V_mean - s * T_mean

    # Compute the average value of the linear regression
    a = 0.5 * s * (t[0] + t[-1]) + b

    return s, a

def quantize(value, breakpoints):
    """
    Quantizes a given value into intervals defined by breakpoints.
    value: Value to quantize
    breakpoints: List of breakpoints defining the quantization intervals
    Returns:
        - The index of the interval the value belongs to
    """
    for i, bp in enumerate(breakpoints):
        if value < bp:
            return i
    return len(breakpoints)


##########################################################################################################################
######################################################### CLASS ##########################################################
##########################################################################################################################

class SAX_transform:
    def __init__(self, series, num_segments=4):
        """
        Initializes the SAX_transforme class with the specified type and parameters.

        Args:
            segments (int): The number of segments for Piecewise Aggregate Approximation (PAA).
        """
        self.num_segments = num_segments
        self.series = series
        self.normalized_series = (self.series - np.mean(self.series)) / np.std(self.series)
        self.segments = self.segment_time_series(self.series)
        self.paa = self.calculate_paa(self.series)

    def segment_time_series(self, ts):
        """
        Segments the time series into `num_segments` equal-sized parts.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
        Returns:
            list: A list of segments (each segment is a sub-array of the time series).
        """
        segment_length = len(ts) // self.num_segments
        segments = []
        for i in range(self.num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length if i != self.num_segments - 1 else len(ts)
            segments.append(ts[start_idx:end_idx])
        return segments

    def calculate_paa(self, ts):
        """
        Perform Piecewise Aggregate Approximation (PAA) on a time series.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
        Returns:
            list: PAA representation as the mean values of each segment.
        """
        segments = self.segment_time_series(ts)
        paa_representation = [np.mean(segment) for segment in segments]
        return paa_representation

    ###################################### SAX ######################################

    def calculate_sax(self, alphabet_size):
        """
        Compute the SAX representation of a time series.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
            alphabet_size (int): Number of symbols in the SAX alphabet.
        Returns:
            str: SAX representation as a string of symbols.
        """
        # Step 1: Perform PAA
        paa_representation = self.calculate_paa(self.normalized_series)

        # Step 3: Determine breakpoints based on Gaussian distribution
        breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

        # Step 4: Map PAA coefficients to symbols
        sax_representation = ''
        for value in paa_representation:
            for i, bp in enumerate(breakpoints):
                if value < bp:
                    sax_representation += chr(97 + i)  # Map to 'a', 'b', ...
                    break
            else:
                sax_representation += chr(97 + alphabet_size - 1)  # Last symbol

        return sax_representation

    ###################################### 1d-SAX ######################################

    def transf_1d_sax(self, Na, Ns):
        """
        Computes the 1d-SAX representation of a given time series.
        time_series: The input time series (array of values)
        L: Length of each segment
        Na: Number of quantization levels for average values
        Ns: Number of quantization levels for slope values
        Returns:
            - A list of symbols representing the 1d-SAX representation
        """
        n = len(self.series)
        segment_size = n // self.segments
        segments = [self.series[i * self.segments:(i + 1) * self.segments] for i in range(segment_size)]
        
        # Compute breakpoints for average and slope values
        avg_breakpoints = norm.ppf(np.linspace(0, 1, Na + 1)[1:-1], loc=0, scale=1)
        slope_variance = 0.03 / self.segments
        slope_breakpoints = norm.ppf(np.linspace(0, 1, Ns + 1)[1:-1], loc=0, scale=np.sqrt(slope_variance))

        symbols = []

        for _, segment in enumerate(segments):
            t = np.arange(len(segment))
            v = np.array(segment)

            s, a = compute_linear_regression(t, v)

            # Quantize the average and slope values
            avg_symbol = quantize(a, avg_breakpoints)
            slope_symbol = quantize(s, slope_breakpoints)

            # Combine the quantized symbols into a single symbol
            combined_symbol = (avg_symbol << int(np.log2(Ns))) | slope_symbol
            symbols.append(combined_symbol)

        return symbols

    ###################################### TVA ######################################

    def calculate_trends(self):
        """
        Compute the trend-based approximation of a time series using least squares.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
        Returns:
            str: Trend representation as a string of 'U', 'D', or 'S'.
        """

        trend_representation = ''

        for segment in self.segments:
            x = np.arange(len(segment))
            y = np.array(segment)
            # Fit a line using least squares
            coeffs = np.polyfit(x, y, 1)  # Linear fit (degree=1)
            slope = coeffs[0]

            # Determine trend character
            if slope > 0:
                trend_representation += 'U'  # Upward trend
            elif slope < 0:
                trend_representation += 'D'  # Downward trend
            else:
                trend_representation += 'S'  # Stable trend

        return trend_representation
    
    def calculate_tva(self, alphabet_size):
        """
        Compute the TVA representation by combining SAX and trend-based approximations.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
            alphabet_size (int): Number of symbols in the SAX alphabet.
        Returns:
            str: TVA representation as a mixed string of SAX values (lowercase) and trends (uppercase).
        """
        # Calculate SAX representation
        sax_rep = self.calculate_sax(alphabet_size)

        # Calculate trend representation
        trend_rep = self.calculate_trends()

        # Combine SAX and trend representations
        tva_representation = ''.join(
            trend_rep[i] + sax_rep[i] for i in range(len(sax_rep))
        )

        return tva_representation
    def mindist(self, sax1, sax2, time_series_length):
        """
        Computes the MINDIST (minimum distance) between two SAX representations.

        Args:
            sax1 (str): First SAX representation.
            sax2 (str): Second SAX representation.
            time_series_length (int): The length of the original time series.

        Returns:
            float: The MINDIST between the two SAX representations.
        """
        if len(sax1) != len(sax2):
            raise ValueError("SAX representations must have the same length.")

        def get_breakpoints(alphabet_size):
            return stats.norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])

        breakpoints = [-np.inf] + list(get_breakpoints(self.alphabet_size)) + [np.inf]

        distance = 0
        for symbol1, symbol2 in zip(sax1, sax2):
            if symbol1 != symbol2:
                idx1 = "abcdefghijklmnopqrstuvwxyz".index(symbol1)
                idx2 = "abcdefghijklmnopqrstuvwxyz".index(symbol2)
                lower_bound = min(breakpoints[idx1 + 1], breakpoints[idx2 + 1])
                upper_bound = max(breakpoints[idx1], breakpoints[idx2])
                distance += (upper_bound - lower_bound) ** 2

        return np.sqrt(distance * (time_series_length / len(sax1)))









