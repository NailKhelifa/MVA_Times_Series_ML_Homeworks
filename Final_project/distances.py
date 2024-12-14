import numpy as np
from scipy.stats import norm
import pandas as pd

class MINDIST():

    def __init__(self, alphabet_size, ts_length):
        self.alphabet_size = alphabet_size
        self.ts_length = ts_length
        self.lookup_table = self.compute_lookup_table()

    def generate_alphabet(self, alphabet_size):
        """
        Generate an alphabet of given size using consecutive letters.
        
        Parameters:
            alphabet_size (int): The number of letters in the alphabet.
            
        Returns:
            list of str: List of alphabet symbols (e.g., ['a', 'b', 'c', ...]).
        """
        return [chr(i) for i in range(ord('a'), ord('a') + alphabet_size)]

    def compute_lookup_table(self):
        """
        Compute a lookup table for symbol distances using a DataFrame for better indexing.
        
        Parameters:
            alphabet_size (int): The size of the alphabet (e.g., 4, 5, etc.).
            
        Returns:
            pd.DataFrame: A DataFrame representing the lookup table with alphabet letters as indices and columns.
        """
        # Generate alphabet
        alphabet = self.generate_alphabet(self.alphabet_size)
        
        # Compute breakpoints
        breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])  # Skip -inf and +inf
        
        # Initialize the lookup table
        lookup_table = np.zeros((self.alphabet_size, self.alphabet_size))
        
        # Fill the lookup table
        for r in range(self.alphabet_size):
            for c in range(self.alphabet_size):
                if abs(r - c) <= 1:
                    lookup_table[r, c] = 0
                else:
                    beta_max = breakpoints[max(r, c) - 1] if max(r, c) > 0 else float('-inf')
                    beta_min = breakpoints[min(r, c)] if min(r, c) < len(breakpoints) else float('inf')
                    lookup_table[r, c] = beta_max - beta_min
        
        # Convert to DataFrame with alphabet as row and column labels
        return pd.DataFrame(lookup_table, index=alphabet, columns=alphabet)

    def mindist(self, sequence1, sequence2):
        """
        Compute the MINDIST distance between two sequences based on the lookup table.
        
        Parameters:
            sequence1 (list of int): First sequence of symbols (indices into the alphabet).
            sequence2 (list of int): Second sequence of symbols (indices into the alphabet).
            lookup_table (np.ndarray): Precomputed lookup table of distances.
            
        Returns:
            float: The MINIST distance between the two sequences.
        """

        lookup_table = self.lookup_table

        if len(sequence1) != len(sequence2):
            raise ValueError("Sequences must have the same length.")
        
        num_segments = len(sequence1)
        distance_squared = 0
        
        for i in range(num_segments):
            dist = lookup_table.loc[sequence1[i], sequence2[i]]
            distance_squared += dist ** 2
        
        return np.sqrt(distance_squared)*np.sqrt(self.ts_length/num_segments)


class TRENDIST(MINDIST):

    def __init__(self, alphabet_size, ts_length, trend_cardinality=5):
        super().__init__(alphabet_size, ts_length)
        self.trend_cardinality = trend_cardinality

    def generate_angle_breakpoints(self):
        """
        Génère dynamiquement les breakpoints d'angles pour diviser l'intervalle (-90°, 90°) en sous-intervalles égaux.

        Arguments :
            angle_breakpoints_length : int, le nombre de sous-intervalles à créer.

        Retourne :
            list : Une liste contenant les breakpoints d'angles.
        """
        step = 180 / self.trend_cardinality  # Largeur d'un sous-intervalle
        breakpoints = [-90 + i * step for i in range(1, self.trend_cardinality)]
        return breakpoints
    

    def compute_angle_lookup_table(self):
        """
        Compute a lookup table for symbol distances using a DataFrame for better indexing.
        
        Parameters:
            alphabet_size (int): The size of the alphabet (e.g., 4, 5, etc.).
            
        Returns:
            pd.DataFrame: A DataFrame representing the lookup table with alphabet letters as indices and columns.
        """
        # Generate alphabet
        alphabet = self.generate_alphabet(self.trend_cardinality)
        
        # Compute breakpoints
        angle_breakpoints = self.generate_angle_breakpoints()
        
        # Initialize the lookup table
        lookup_table = np.zeros((self.trend_cardinality, self.trend_cardinality))
        
        # Fill the lookup table
        for r in range(self.trend_cardinality):
            for c in range(self.trend_cardinality):
                if abs(r - c) <= 1:
                    lookup_table[r, c] = 0
                else:
                    theta_max = angle_breakpoints[max(r, c) - 1] if max(r, c) > 0 else float('-inf')
                    theta_min = angle_breakpoints[min(r, c)] if min(r, c) < len(angle_breakpoints) else float('inf')
                    lookup_table[r, c] = np.tan(theta_max - theta_min)
        
        # Convert to DataFrame with alphabet as row and column labels
        return pd.DataFrame(lookup_table, index=alphabet, columns=alphabet)
    
    def tsax_mindist(self, tsx1, tsx2):
        """
        Calculates the TSX distance (TSX_DIST) between two time series representations tsx1 and tsx2.

        Parameters:
            tsx1, tsx2: list of SAX representations of two time series, each element is a tuple (mean_symbol, trend_symbol)
            ts_length: int, original length of the time series
            trend_cardinality: int, the trend cardinality (default=5)

        Returns:
            float: the TSX distance between tsx1 and tsx2.
        """
        n = self.ts_length
        w = len(tsx1)
        compression_ratio = n / w

        mean_distance = 0
        trend_distance = 0

        mindist = MINDIST(self.alphabet_size, self.ts_length)

        mindist_lookup_table = mindist.compute_lookup_table()
        angle_lookup_table = self.compute_angle_lookup_table()

        for i in range(w // 4):
            mean_distance += mindist_lookup_table.loc[tsx1[4*i], tsx2[i][4*i]] ** 2
            for j in range(1, 4):  # Compute trend distances for each component of the trend
                trend_distance += angle_lookup_table.loc[tsx1[4*i + j], tsx2[4*i + j]] ** 2

        tsx_dist = np.sqrt(compression_ratio * mean_distance + ((compression_ratio / w) ** 2) * trend_distance)
        
        return tsx_dist
