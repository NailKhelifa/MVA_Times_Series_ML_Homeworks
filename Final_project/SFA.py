import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm
import math
from pyts.approximation import SymbolicFourierApproximation
from pyts.transformation import BOSS
from sklearn.utils import check_array

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


###################################################################################################################################
################################################### BOSS #########################################################################
###################################################################################################################################

class BOSS_class: 

    def __init__(self, X_train, word_size, window_size, alphabet_size, X_test = None, strategy='quantile'):
        self.alphabet_size = alphabet_size
        self.word_size = word_size
        self.window_size = window_size
        self.X_train = X_train
        self.X_test = X_test
        self.strategy = strategy

        self.BOSS = BOSS(word_size=self.word_size, n_bins=self.alphabet_size, window_size=self.window_size, strategy=self.strategy)
        self.symbolic_data = None
        self.symbolic_data_test = None

    def symbolize_BOSS(self): 

        def get_words_from_boss_representation(boss_representation, word_size):
            """
            Convertit la représentation BOSS en mots symboliques.

            Parameters:
                boss_representation (sparse matrix): Représentation BOSS.
                word_size (int): Taille du mot symbolique.

            Returns:
                words (list): Liste des mots symboliques.
            """
            data = boss_representation.toarray().flatten()  # Convertir la matrice sparse en array dense
            words = [data[i:i + word_size] for i in range(0, len(data), word_size)]  # Regrouper en mots
            return words
        
        X_train_BOSS = self.BOSS.fit_transform(self.X_train)
        self.symbolic_data = get_words_from_boss_representation(X_train_BOSS, self.word_size)

        if self.X_test is not None:
            X_test_BOSS = self.BOSS.transform(self.X_test)
            self.symbolic_data_test = get_words_from_boss_representation(X_test_BOSS, self.word_size)
        
    def boss_dist(x, y):
        ## EXTRACTED FROM PYTS PACKAGE
        """Return the BOSS distance between two arrays.

        Parameters
        ----------

        x : array-like, shape = (n_timestamps,)
            First array.

        y : array-like, shape = (n_timestamps,)
            Second array.

        Returns
        -------
        dist : float
            The BOSS distance between both arrays.

        Notes
        -----
        The BOSS metric is defined as

        .. math::

            BOSS(x, y) = \sum_{\substack{i=1\\ x_i > 0}}^n (x_i - y_i)^2

        where :math:`x` and :math:`y` are vectors of non-negative integers.
        The BOSS distance is not a distance metric as it neither satisfies the
        symmetry condition nor the triangle inequality.

        References
        ----------
        .. [1] P. Schäfer, "The BOSS is concerned with time series classification
            in the presence of noise". Data Mining and Knowledge Discovery,
            29(6), 1505-1530 (2015).

        Examples
        --------
        >>> from pyts.metrics import boss
        >>> x = [0, 5, 5, 3, 4, 5]
        >>> y = [3, 0, 0, 0, 8, 0]
        >>> boss(x, y)
        10.0
        >>> boss(y, x)
        5.0

        """
        x = check_array(x, ensure_2d=False, dtype='float64')
        y = check_array(y, ensure_2d=False, dtype='float64')
        if x.ndim != 1:
            raise ValueError("'x' must a one-dimensional array.")
        if y.ndim != 1:
            raise ValueError("'y' must a one-dimensional array.")
        if x.shape != y.shape:
            raise ValueError("'x' and 'y' must have the same shape.")

        non_zero_idx = ~np.isclose(x, np.zeros_like(x), rtol=1e-5, atol=1e-8)
        return np.sqrt(np.sum((x[non_zero_idx] - y[non_zero_idx]) ** 2))



    
    
        