from SAX_transf import SAX_transform
from distances import MINDIST
import numpy as np
import pandas as pd
from collections import Counter
##Packages KNN_DTW
from dtw import dtw
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc



class SYMBOLS():

    def __init__(self, X_train, X_test, method='SAX', k=1, num_segments=20, alphabet_size=16):
        self.k = k # number of neighbors for the prediction 
        self.method = method
        self.X_train = X_train
        self.X_test = X_test
        self.num_train_samples, self.train_ts_length = X_train.shape
        self.num_test_samples, self.test_ts_length = X_test.shape
        self.alphabet_size = alphabet_size
        self.num_segments = num_segments
        # extract train labels (y_train) and train features (x_train) from X_train
        self.X_train.columns = list(self.X_train.columns[:-1]) + ['label']
        x_train, y_train = self.X_train.iloc[:, :-1], self.X_train["label"]
        self.x_train = x_train
        self.y_train = y_train.replace(-1, 0)
        # extract test labels (y_test) and test features (x_test) from X_test
        self.X_test.columns = list(self.X_test.columns[:-1]) + ['label']
        x_test, y_test = self.X_test.iloc[:, :-1], self.X_test["label"]
        self.x_test = x_test
        self.y_test = y_test.replace(-1, 0)     

    def symbolize(self):   
        # Initialiser le DataFrame pour les symboles
        train_symbolic_data = pd.DataFrame(np.zeros((self.num_train_samples, 1)))
        test_symbolic_data = pd.DataFrame(np.zeros((self.num_test_samples, 1)))

        ## symbolize the train time series 
        for i in range(self.num_train_samples):
            ts = self.x_train.iloc[i, :]
            
            # Transformation SAX
            sax_trans = SAX_transform(ts, self.num_segments, self.alphabet_size)
            if self.method == "SAX":
                symbolic_seq = sax_trans.calculate_sax()
                # Vérifier que la longueur de symbolic_seq correspond à num_segments
                if len(symbolic_seq) != self.num_segments:
                    raise ValueError(f"La longueur de symbolic_seq ({len(symbolic_seq)}) "
                                    f"ne correspond pas à num_segments ({self.num_segments})")
            elif self.method == "ESAX":
                symbolic_seq = sax_trans.calculate_esax()


            

            train_symbolic_data.iloc[i, :] = symbolic_seq

        ## symbolize the test time series 
        for i in range(self.num_test_samples):
            ts = self.x_test.iloc[i, :]
            
            # Transformation SAX
            sax_trans = SAX_transform(ts, self.num_segments, self.alphabet_size)
            if self.method == "SAX":
                symbolic_seq = sax_trans.calculate_sax()
                # Vérifier que la longueur de symbolic_seq correspond à num_segments
                if len(symbolic_seq) != self.num_segments:
                    raise ValueError(f"La longueur de symbolic_seq ({len(symbolic_seq)}) "
                                    f"ne correspond pas à num_segments ({self.num_segments})")
            elif self.method == "ESAX":
                symbolic_seq = sax_trans.calculate_esax()
            
            

            test_symbolic_data.iloc[i, :] = symbolic_seq

        ## save the symbolized time series as attributes
        self.symbolized_x_train = train_symbolic_data
        self.symbolized_x_test = test_symbolic_data

    def mindist(self, sax1, sax2, ts_length):
        """
        Computes the MINDIST (minimum distance) between two SAX representations.

        Args:
            sax1 (str): First SAX representation.
            sax2 (str): Second SAX representation.
            time_series_length (int): The length of the original time series.

        Returns:
            float: The MINDIST between the two SAX representations.
        """
  
        mindist = MINDIST(self.alphabet_size, ts_length)

        return mindist.mindist(sax1, sax2)
    
    # Prédiction pour un seul point de test
    def _predict(self):

        predictions = []
        # make a prediction based on k-nn for each symbolized series in the test dataset
        for j in range(self.num_test_samples):
            # Calcul des distances entre x_test et tous les points d'entraînement
            distances = [self.mindist(self.symbolized_x_test.iloc[j][0], self.symbolized_x_train.iloc[i][0], self.train_ts_length) for i in range(self.num_train_samples)]
            
            # Trier les distances et obtenir les indices des k plus proches voisins
            k_indices = np.argsort(distances)[:self.k]
            
            # Récupérer les classes des k plus proches voisins
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Classification : renvoyer la classe la plus fréquente
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        self.predictions = predictions
        # Calcul de l'exactitude : proportion de bonnes prédictions
        self.accuracy = np.mean(predictions == self.y_test)
        
        print(f"Accuracy:{self.accuracy}")


