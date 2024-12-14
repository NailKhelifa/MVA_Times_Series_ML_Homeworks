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

    def __init__(self, dataset, num_segments, alphabet_size, pen_factor = None, mean_or_slope = 'mean'):
        self.dataset = dataset
        self.num_segments = num_segments
        self.alphabet_size = alphabet_size
        self.num_samples = dataset.shape[0]
        self.sample_size = dataset.shape[1]
        self.pen_factor = pen_factor
        self.mean_or_slope = mean_or_slope
        self.mts_bkps_ = None

        ## On initialise les données symboliques à None
        self.symbolic_data = None
    def segmentation_adaptive(self, *args, **kwargs):
        """In case of multivariate adaptive segmentation, get the list of
        multivariate breakpoints."""

        list_of_signals = self.dataset
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

        for series in self.dataset:  
            
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

        # Étape 4 : Conversion des indices en symboles alphabétiques
        symbolic_dataset_str = [
            [chr(64 + num) for num in symbolic_series] 
            for symbolic_series in symbolic_dataset
        ]

        self.symbolic_data = symbolic_dataset_str

        return self.symbolic_data

    def reconstruction_from_ASTRIDE(self, ASTRIDE_symbols): 

