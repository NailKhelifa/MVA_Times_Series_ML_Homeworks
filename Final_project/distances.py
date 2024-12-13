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

