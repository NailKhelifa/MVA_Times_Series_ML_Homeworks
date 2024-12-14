import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm
import math
import ruptures as rpt
#from sklearn.base import BaseEstimator
#from sklearn.utils import Bunch


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
######################################################### SAX ##########################################################
##########################################################################################################################

class SAX_transform:

    def __init__(self, series, num_segments, alphabet_size):
        """
        Initializes the SAX_transforme class with the specified type and parameters.

        Args:
            segments (int): The number of segments for Piecewise Aggregate Approximation (PAA).
        """
        self.num_segments = num_segments
        self.alphabet_size = alphabet_size
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

    def esax_paa_min_max(self, ts):
        """
        Perform Piecewise Aggregate Approximation (PAA) on a time series.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
        Returns:
            list: PAA representation as the mean values of each segment.
        """
        segments = self.segment_time_series(ts)
        esax_paa_representation = [[np.min(segment), np.mean(segment), np.max(segment)] for segment in segments]

        return esax_paa_representation

    def tsax_paa(self, ts):
        """
        Perform Piecewise Aggregate Approximation (PAA) on a time series.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
        Returns:
            list: PAA representation as the mean values of each segment.
        """
        segments = self.segment_time_series(ts)
        tsax_paa = []

        for segment in segments:

            angles_tuple = segment_to_angles(segment)
            tsax_paa.append([np.mean(segment), angles_tuple[0], angles_tuple[1], angles_tuple[2]])

        return tsax_paa

    ###################################### SAX ######################################

    def calculate_sax(self):
        """
        Compute the ESAX representation of a time series.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
        Returns:
            str: SAX representation as a string of symbols.
        """
        # Step 1: Perform PAA
        paa_representation = self.calculate_paa(self.normalized_series)

        # Step 3: Determine breakpoints based on Gaussian distribution
        breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])

        # Step 4: Map PAA coefficients to symbols
        sax_representation = ''
        for value in paa_representation:
            for i, bp in enumerate(breakpoints):
                if value < bp:
                    sax_representation += chr(97 + i)  # Map to 'a', 'b', ...
                    break
            else:
                sax_representation += chr(97 + self.alphabet_size - 1)  # Last symbol

        return sax_representation

    ###################################### ESAX #####################################
    def calculate_esax(self):
        """
        Compute the SAX representation of a time series.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
        Returns:
            str: SAX representation as a string of symbols.
        """
        # Step 1: Perform PAA
        paa_representation = self.esax_paa_min_max(self.normalized_series)

        # Step 3: Determine breakpoints based on Gaussian distribution
        breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])

        # Step 4: Map PAA coefficients to symbols
        esax_representation = ''
        for value in paa_representation:
            for value2 in value:
                for i, bp in enumerate(breakpoints):
                    if value2 < bp:
                        esax_representation += chr(97 + i)  # Map to 'a', 'b', ...
                        break
                else:
                    esax_representation += chr(97 + self.alphabet_size - 1)  # Last symbol

        return esax_representation

    def reconstruction_from_sax(self, sax_representation, option = None):
            """
            Reconstructs an approximation of the time series from its SAX representation.
            Args:
                sax_representation : SAX representation as a string of symbols.
                num_segments : Number of segments.
            Returns:
                array-like: Reconstructed time series.
            """
            # On détermine les breakpoints
            breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])

            # On reconstruit la série temporelle -> on prend la moyenne des breakpoints correspondant à chaque symbole
            paa_values = []
            if option == "esax":
                sax_representation = sax_representation[1::3]
                print(f'Voici la nouvelle représentation ESAX: {sax_representation} et sa taille{len(sax_representation)}')

            for symbol in sax_representation:
                if ord(symbol) == 97 + self.alphabet_size - 1:
                    paa_values.append(breakpoints[-1])
                elif ord(symbol) == 97:
                    paa_values.append(breakpoints[0])
                else:
                    paa_values.append(np.mean([breakpoints[ord(symbol) - 97], breakpoints[ord(symbol) - 98]]))

            # On reconstruit la série temporelle
            segment_size = len(self.series) // self.num_segments
            reconstructed_series = np.repeat(paa_values[:-1], len(self.series) // self.num_segments)
            last_segment_size = len(self.series) - segment_size * (self.num_segments - 1)
            reconstructed_series = np.append(reconstructed_series, paa_values[-1] * np.ones(last_segment_size))

            return reconstructed_series
    ###################################### TSAX #####################################

    def calculate_tsax(self, angle_breakpoint_alphabet_size):
        """
        Compute the SAX representation of a time series.
        Args:
            ts (array-like): Input time series.
            num_segments (int): Number of segments.
        Returns:
            str: SAX representation as a string of symbols.
        """
        # Step 1: Perform PAA
        paa_representation = self.tsax_paa(self.normalized_series)

        # Step 3: Determine breakpoints based on Gaussian distribution
        breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
        angles_breakpoints = generate_angle_breakpoints(angle_breakpoint_alphabet_size)

        # Step 4: Map PAA coefficients to symbols
        tsax_representation = ''
        for value in paa_representation:
            # compute the symbol for the first element in the list (mean, corresponding to SAX)
            for i, bp in enumerate(breakpoints):
                if value[0] < bp:
                    tsax_representation += chr(97 + i)  # Map to 'a', 'b', ...
                    break
            else:
                tsax_representation += chr(97 + self.alphabet_size - 1)  # Last symbol

            # compute the tree following symbols for the three angle-linked elements 
            angles = value[1:]
            three_angles_symbols = map_angles_to_symbols(angles, angle_breakpoint_alphabet_size)
            tsax_representation += three_angles_symbols


        return tsax_representation
    
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
        segment_size = n // self.num_segments
        segments = self.segment_time_series(self.series)
        
        # Compute breakpoints for average and slope values
        avg_breakpoints = norm.ppf(np.linspace(0, 1, Na + 1)[1:-1], loc=0, scale=1)
        slope_variance = 0.03 / self.num_segments
        slope_breakpoints = norm.ppf(np.linspace(0, 1, Ns + 1)[1:-1], loc=0, scale=np.sqrt(slope_variance))

        OneD_SAX = []

        for _, segment in enumerate(segments):
            t = np.arange(len(segment))
            v = np.array(segment)

            s, a = compute_linear_regression(t, v)

            # Quantize the average and slope values
            avg_symbol = quantize(a, avg_breakpoints)
            slope_symbol = quantize(s, slope_breakpoints)

            # Combine the quantized symbols into a single symbol
            combined_symbol = (avg_symbol << int(np.log2(Ns))) | slope_symbol
            OneD_SAX.append(combined_symbol)

        return OneD_SAX

    def reconstruct_from_1d_sax(self, OneD_SAX, Na, Ns):
        """
        Reconstructs an approximation of the time series from its 1d-SAX representation.
        
        OneD_SAX: List of symbols representing the 1d-SAX representation.
        Na: Number of quantization levels for average values.
        Ns: Number of quantization levels for slope values.
        
        Returns:
            - A list representing the reconstructed time series.
        """
        # On recalcule les breakpoints
        avg_breakpoints = norm.ppf(np.linspace(0, 1, Na + 1)[1:-1], loc=0, scale=1)
        slope_variance = 0.03 / self.num_segments
        slope_breakpoints = norm.ppf(np.linspace(0, 1, Ns + 1)[1:-1], loc=0, scale=np.sqrt(slope_variance))

        reconstructed_series = []
        segment_size = len(self.series) // self.num_segments

        for symbol in OneD_SAX:

            # Par construction de 1d-SAX, on peut retrouver les symboles de la moyenne et de la pente
            avg_symbol = symbol >> int(np.log2(Ns))
            slope_symbol = symbol & (Ns - 1)

            if avg_symbol == 0: 
                approx_avg = avg_breakpoints[0]
            elif avg_symbol == Na - 1:
                approx_avg = avg_breakpoints[-1]
            else:
                approx_avg = (avg_breakpoints[avg_symbol - 1] + avg_breakpoints[avg_symbol]) / 2 

            if slope_symbol == 0:
                approx_slope = slope_breakpoints[0]
            elif slope_symbol == Ns - 1:
                approx_slope = slope_breakpoints[-1]
            else:
                approx_slope = (slope_breakpoints[slope_symbol - 1] + slope_breakpoints[slope_symbol]) / 2 

            segment = [approx_avg + approx_slope * (t - 1) for t in range(1, segment_size + 1)]
            reconstructed_series.extend(segment)

        return reconstructed_series

    
##########################################################################################################################
###################################################### TSAX utils ########################################################
##########################################################################################################################

def find_key_points(time_series):
    """
    Détecte les points clés (start, MP, MD, end) dans une série temporelle.
    - Start : Premier point.
    - End : Dernier point.
    - MP : Point avec la plus grande distance verticale au-dessus de la ligne de tendance.
    - MD : Point avec la plus grande distance verticale en dessous de la ligne de tendance.

    Arguments :
        time_series : list of float, la série temporelle (les valeurs y au fil du temps).

    Retourne :
        list : Une liste de quatre points clés sous la forme [(x_start, y_start), (x_MP, y_MP), (x_MD, y_MD), (x_end, y_end)].
    """
    # Étape 1 : Convertir la série temporelle en points (x, y)
    points = [(i, y) for i, y in enumerate(time_series)]  # i = temps, y = valeur
    
    # Étape 2 : Initialiser les points start et end
    start = points[0]
    end = points[-1]
    
    # Étape 3 : Calculer la ligne de tendance entre start et end
    def vertical_distance(p, start, end):
        """
        Calcule la distance verticale (VD) d'un point p à la ligne de tendance entre start et end.
        """
        x1, y1 = start
        x2, y2 = end
        x, y = p

        # Équation de la droite (ligne de tendance)
        if x2 - x1 == 0:  # Éviter la division par zéro pour une ligne verticale
            return abs(x - x1)
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        y_trend = slope * x + intercept  # Valeur prédite par la ligne de tendance
        
        return y - y_trend  # Distance verticale (positive si au-dessus, négative si en dessous)
    
    # Étape 4 : Rechercher MP (Maximum Peak) et MD (Maximum Dip)
    max_positive_distance = float('-inf')
    max_negative_distance = float('inf')
    MP = None
    MD = None

    for point in points[1:-1]:  # Ignorer start et end
        distance = vertical_distance(point, start, end)
        if distance > max_positive_distance:
            max_positive_distance = distance
            MP = point
        if distance < max_negative_distance:
            max_negative_distance = distance
            MD = point

    # Étape 5 : Gérer les cas où MP ou MD n'existent pas (grâce à une initialisation par défaut)
    if MP is None:
        MP = start  # Par défaut, MP est le point de départ si aucun pic n'existe
    if MD is None:
        MD = start  # Par défaut, MD est le point de départ si aucune dépression n'existe

    # Étape 6 : Retourner les quatre points clés
    return [start, MP, MD, end]

def calculate_trend_angles(points):
    """
    Calcule les trois angles de tendance entre les quatre points [start, MP, MD, end].

    Arguments :
        points : list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] - Les quatre points clés.
                Les points doivent être dans l'ordre : start, MP, MD, end.

    Retourne :
        tuple : Les trois angles de tendance (en degrés) définis par les trois segments reliant les quatre points.
    """
    def angle_between_points(p1, p2):
        """
        Calcule l'angle (en radians) entre deux points p1 et p2 par rapport à l'axe horizontal.
        """
        x1, y1 = p1
        x2, y2 = p2
        return math.atan2(y2 - y1, x2 - x1)  # Angle par rapport à l'horizontale

    # Extraire les quatre points clés
    start, MP, MD, end = points
    
    # Calcul des angles des trois segments
    angle1 = angle_between_points(start, MP)  # Angle du segment start -> MP
    angle2 = angle_between_points(MP, MD)    # Angle du segment MP -> MD
    angle3 = angle_between_points(MD, end)   # Angle du segment MD -> end

    # Calcul des différences d'angles entre les segments consécutifs
    trend_angle1 = math.degrees(angle2 - angle1)  # Angle entre le premier et le deuxième segment
    trend_angle2 = math.degrees(angle3 - angle2)  # Angle entre le deuxième et le troisième segment

    # Normaliser les angles pour qu'ils soient compris entre -180° et 180°
    trend_angle1 = (trend_angle1 + 180) % 360 - 180
    trend_angle2 = (trend_angle2 + 180) % 360 - 180

    # Ajouter les deux angles de tendance et l'angle absolu final
    trend_angle3 = math.degrees(angle3)  # L'angle final absolu (MD -> end par rapport à l'horizontale)

    return (trend_angle1, trend_angle2, trend_angle3)

def segment_to_angles(segment_ts):
    
    four_points_list = find_key_points(segment_ts)
    angles_tuple = calculate_trend_angles(four_points_list)

    return angles_tuple

def generate_angle_breakpoints(angle_breakpoints_length):
    """
    Génère dynamiquement les breakpoints d'angles pour diviser l'intervalle (-90°, 90°) en sous-intervalles égaux.

    Arguments :
        angle_breakpoints_length : int, le nombre de sous-intervalles à créer.

    Retourne :
        list : Une liste contenant les breakpoints d'angles.
    """
    step = 180 / angle_breakpoints_length  # Largeur d'un sous-intervalle
    breakpoints = [-90 + i * step for i in range(1, angle_breakpoints_length)]
    return breakpoints

def map_angles_to_symbols(angles, angle_breakpoints_length):
    """
    Mappe les angles aux symboles en utilisant des breakpoints dynamiquement générés.

    Arguments :
        angles : tuple de trois angles (angle1, angle2, angle3) en degrés.
        angle_breakpoints_length : int, nombre d'intervalles pour les breakpoints.

    Retourne :
        str : Une chaîne de longueur 3 composée des symboles ('a', 'b', ..., selon le nombre d'intervalles).
    """
    # Génération des breakpoints dynamiquement
    breakpoints = generate_angle_breakpoints(angle_breakpoints_length)
    symbols = [chr(ord('a') + i) for i in range(angle_breakpoints_length)]  # Générer les symboles dynamiquement

    def map_angle(angle):
        """
        Mappe un angle unique à un symbole selon les breakpoints.
        """
        for i in range(len(breakpoints)):
            if angle < breakpoints[i]:
                return symbols[i]
        return symbols[-1]  # Retourne le dernier symbole si l'angle dépasse tous les breakpoints

    # Appliquer la fonction à chaque angle du tuple
    symbol1 = map_angle(angles[0])
    symbol2 = map_angle(angles[1])
    symbol3 = map_angle(angles[2])

    # Retourner la chaîne des symboles
    return f"{symbol1}{symbol2}{symbol3}"