import numpy as np
import scipy.stats as stats # for the breakpoints in SAX
import pandas as pd
from scipy.stats import norm
import math
import ruptures as rpt


##########################################################################################################################
###################################################### ASTRIDE ###########################################################
##########################################################################################################################




class ASTRIDE_transf:

    def __init__(self, X_train,  num_segments, alphabet_size, X_test = None, pen_factor = None, mean_or_slope = 'mean'):
        self.X_train = X_train
        self.X_test = X_test
        self.num_segments = num_segments
        self.alphabet_size = alphabet_size
        self.num_samples = X_train.shape[0]
        self.sample_size = X_train.shape[1]
        self.pen_factor = pen_factor
        self.mean_or_slope = mean_or_slope
        self.mts_bkps_ = None

        ## On initialise les données symboliques à None
        self.symbolic_data = None
        self.symbolic_data_test = None
        self.segmentation_adaptive()
        self.transform_adaptive(self.X_train)


    def segmentation_adaptive(self, *args, **kwargs):
        """In case of multivariate adaptive segmentation, get the list of
        multivariate breakpoints."""

        ## On fait la segmentation sur le train
        list_of_signals = self.X_train
        # `list_of_signals` must of shape (n_signals, n_samples)
        if len(np.shape(list_of_signals)) == 3:
            X = list_of_signals[:,:,0]
        else:
            X = list_of_signals
        self.mts_bkps_ = self.transform_adaptive(np.transpose(X))


    def transform_adaptive(self, signal):
        """Return change-points indexes for mean or slope shifts."""

        if self.mean_or_slope == "slope":
            # BottomUp for slope
            algo = rpt.BottomUp(model="clinear", jump=1).fit(signal)
        elif self.mean_or_slope == "mean":
            # Dynp for mean
            algo = rpt.KernelCPD(kernel="linear", jump=1).fit(signal)

        if self.num_segments is not None:
            n_bkps = self.num_segments - 1
            bkps = algo.predict(n_bkps=n_bkps)
        elif self.pen_factor is not None:
            pen_value = self.get_penalty_value(signal)
            bkps = algo.predict(pen=pen_value)

        return bkps
    

    def get_penalty_value(self, signal):
        """Return penalty value for a single signal."""
        n_samples = signal.shape[0]
        return self.pen_factor * np.log(n_samples)
    
    
    def _ASTRIDE_symbolize(self):
        
        #Calcul des moyennes pour tous les segments de toutes les séries
        all_segments_means = []
        symbolic_dataset = []

        all_segments_means_test = []
        symbolic_dataset_test = []

        for series in self.X_train:  
            
            segments_means = [np.mean(series[self.mts_bkps_[i]:self.mts_bkps_[i+1]]) 
                            for i in range(len(self.mts_bkps_) - 1)]
            all_segments_means.extend(segments_means)  
            symbolic_dataset.append(segments_means)  


        #Calcul des quantiles empiriques sur toutes les moyennes
        quantiles = np.quantile(all_segments_means, np.linspace(0, 1, self.alphabet_size + 1)[1:-1])

        #Binning des segments selon les quantiles
        symbolic_dataset = [
            [np.digitize(mean, quantiles) for mean in segments] 
            for segments in symbolic_dataset
        ]
        
        # Conversion des indices en symboles alphabétiques
        symbolic_dataset_str = [
            "".join([chr(65 + num) for num in symbolic_series]) 
            for symbolic_series in symbolic_dataset
        ]

    
        self.symbolic_data = symbolic_dataset_str

        if self.X_test is not None:
            for series in self.X_test:  
                
                segments_means = [np.mean(series[self.mts_bkps_[i]:self.mts_bkps_[i+1]]) 
                                for i in range(len(self.mts_bkps_) - 1)]
                all_segments_means_test.extend(segments_means)  
                symbolic_dataset_test.append(segments_means)
            
            symbolic_dataset_test = [
            [np.digitize(mean, quantiles) for mean in segments]
            for segments in symbolic_dataset_test
            ]

            symbolic_dataset_str_test = [
            "".join([chr(65 + num) for num in symbolic_series])
            for symbolic_series in symbolic_dataset_test
            ]

            self.symbolic_data_test = symbolic_dataset_str_test

        return self.symbolic_data
    

    def reconstruction_from_ASTRIDE(self, ASTRIDE_symbols):
        reconstructed_dataset = []

        # Calcul des moyennes associées aux quantiles
        quantiles = np.quantile(
            [np.mean(series[self.mts_bkps_[i]:self.mts_bkps_[i+1]]) 
            for series in self.X_train 
            for i in range(len(self.mts_bkps_) - 1)],
            np.linspace(0, 1, self.alphabet_size)
        )

        # Reconstruire chaque série
        for series_idx, symbolic_series in enumerate(ASTRIDE_symbols):
            reconstructed_series = np.zeros(len(self.X_train[0]))
            for i, symbol in enumerate(symbolic_series):
                # Trouver la valeur moyenne associée au symbole
                segment_mean = quantiles[ord(symbol) - 65]  

                # Vérification des limites de segment
                if i >= len(self.mts_bkps_) - 1:
                    break

                start = self.mts_bkps_[i]
                end = self.mts_bkps_[i + 1]
                segment_length = end - start

                # Remplir le segment reconstruit avec la valeur moyenne
                reconstructed_series[start:end] = segment_mean

            reconstructed_dataset.append(reconstructed_series)

        return np.array(reconstructed_dataset)

    
    def calculate_dged(self, ASTRIDE_symbols_1, ASTRIDE_symbols_2):
        
        # Calcul des quantiles
        quantiles_1 = np.quantile(
            [np.mean(series[self.mts_bkps_[i]:self.mts_bkps_[i+1]]) 
            for series in self.X_train 
            for i in range(len(self.mts_bkps_) - 1)],
            np.linspace(0, 1, self.alphabet_size)
        )

        ## On prend en compte la taille des segments

        segments_size = [self.mts_bkps_[i+1] - self.mts_bkps_[i] for i in range(len(self.mts_bkps_) - 1)] 
        nb_symbols_by_seg = np.array(segments_size) // min(segments_size)

        symbolic_serie1 = []
        symbolic_serie2 = []

        for i in range(len(segments_size)):
            symbolic_serie1.extend([ASTRIDE_symbols_1[i]] * nb_symbols_by_seg[i])
            symbolic_serie2.extend([ASTRIDE_symbols_2[i]] * nb_symbols_by_seg[i])

        ## Ici on calcule la distance D-GED, en considérant que les deux séries ont le même nombre de segments
        ## On a que le coût de substitution est la distance euclidienne entre les moyennes des segments

        # Calcul de la distance D-GED
        dged = 0

        for i in range(len(symbolic_serie1)):
            symbol_1 = symbolic_serie1[i]
            symbol_2 = symbolic_serie2[i]

            segment_mean_1 = quantiles_1[ord(symbol_1) - 65] 
            segment_mean_2 = quantiles_1[ord(symbol_2) - 65]

            sub_cost = np.sqrt((segment_mean_1 - segment_mean_2)**2)
            dged += sub_cost

        return dged