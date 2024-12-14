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
        self.symbolic_data = np.zeros((self.num_samples, self.num_segments))
        self.pen_factor = pen_factor
        self.mean_or_slope = mean_or_slope
        self.mts_bkps_ = None

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

