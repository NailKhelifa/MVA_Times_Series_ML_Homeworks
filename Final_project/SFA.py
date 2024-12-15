import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm
import math
from pyts.approximation import SymbolicFourierApproximation

##########################################################################################################################
################################################### SFA / BOSS ###########################################################
##########################################################################################################################

class SFA:

    def __init__(self, X_train, num_coefs, alphabet_size, X_test =None, strategy='quantile'):
        self.alphabet_size = alphabet_size
        self.num_coefs = num_coefs
        self.X_train = X_train
        self.X_test = X_test
        self.strategy = strategy

        self.symbolic_data = None
        self.symbolic_data_test = None
        self.sfa = SymbolicFourierApproximation(n_coefs=self.num_coefs,  n_bins=self.alphabet_size, strategy=self.strategy)
        

    def symbolize_SFA(self):
        """
        Symbolize the time series using the SFA algorithm.
        
        Parameters:
            X (np.ndarray): The time series data.
            
        Returns:
            np.ndarray: The symbolized time series.
        """
    
        # Transformer les séries temporelles en représentations symboliques
        X_train_sfa = self.sfa.fit_transform(self.X_train)
        self.symbolic_data = X_train_sfa
        self.bin_edges = self.sfa.bin_edges_
        
        if self.X_test is not None:
            X_test_sfa = self.sfa.transform(self.X_test)
            self.symbolic_data_test = X_test_sfa
        
    def SFA_distance(self, sequence1, sequence2):
        """
        We define a distance, inspired by the Euclidean distance and the initial Paper, between two sequences of symbols.
        
        Parameters:
            sequence1 (np.ndarray): The first sequence.
            sequence2 (np.ndarray): The second sequence.
            
        Returns:
            float: The Euclidean distance between the two sequences.
        """
        if len(sequence1) != len(sequence2):
            raise ValueError("Sequences must have the same length.")
        
        num_segments = len(sequence1)
        distance_squared = 0
        
        for i in range(num_segments):
            if sequence1[i] == sequence2[i]:
                continue
            
            num1 = ord(sequence1[i]) - ord('a')
            num2 = ord(sequence2[i]) - ord('a')

            if num1 == 0: 
                edge1 = [self.bin_edges[i][0]]*2
            elif num1 == self.alphabet_size - 1:
                edge1 = [self.bin_edges[i][-1]]*2
            else: 
                edge1 = [self.bin_edges[i][num1-1], self.bin_edges[i][num1]]
            
            if num2 == 0:
                edge2 = [self.bin_edges[i][0]]*2
            elif num2 == self.alphabet_size - 1:
                edge2 = [self.bin_edges[i][-1]]*2
            else:
                edge2 = [self.bin_edges[i][num2-1], self.bin_edges[i][num2]]

            
            dist = np.min([abs(edge1[0] - edge2[1]), abs(edge1[1] - edge2[0]), abs(edge1[0] - edge2[0]), abs(edge1[1] - edge2[1])])
            distance_squared += dist ** 2
        
        return np.sqrt(distance_squared)
    
    
        