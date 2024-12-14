import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm


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

    ###################################### TVA ######################################

    
##########################################################################################################################
######################################################### SYMBOL ##########################################################
##########################################################################################################################

