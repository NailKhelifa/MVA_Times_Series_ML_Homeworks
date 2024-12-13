import numpy as np
from scipy.integrate import quad
from scipy.spatial.distance import euclidean
from sklearn.tree import DecisionTreeClassifier


class RegressionTree:
    def __init__(self, threshold, X_train, X_test):
        """
        Initialize the classifier with a given threshold for pattern detection.

        Args:
            threshold (float): The threshold distance to detect patterns.
        """
        self.threshold = threshold
        self.patterns = []

        self.X_train = X_train
        self.X_test = X_test
        self.num_train_samples, self.train_ts_length = X_train.shape
        self.num_test_samples, self.test_ts_length = X_test.shape
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
        
        self.clf = DecisionTreeClassifier()

    def piecewise_constant_model(self, signal, n_max):
        """
        Discretize a signal using a piecewise constant model.

        Args:
            signal (np.array): The signal to discretize.
            n_max (int): Maximum number of pieces.

        Returns:
            np.array: Discretized signal.
        """
        n_samples = len(signal)
        discontinuity_points = [0, n_samples]
        model = np.zeros_like(signal, dtype=float)

        for _ in range(n_max):
            max_var_reduction, best_split = 0, None

            for i in range(1, len(discontinuity_points)):
                start, end = discontinuity_points[i - 1], discontinuity_points[i]
                segment = signal[start:end]

                for t in range(start + 1, end):
                    left_var = np.var(signal[start:t]) * (t - start)
                    right_var = np.var(signal[t:end]) * (end - t)

                    total_var_reduction = np.var(segment) * len(segment) - (left_var + right_var)
                    if total_var_reduction > max_var_reduction:
                        max_var_reduction, best_split = total_var_reduction, t

            if best_split:
                discontinuity_points.append(best_split)
                discontinuity_points.sort()

        for i in range(1, len(discontinuity_points)):
            start, end = discontinuity_points[i - 1], discontinuity_points[i]
            model[start:end] = np.mean(signal[start:end])

        return model

    def detect_pattern(self, reference_signal, test_signal):
        """
        Detect whether a pattern exists in a signal.

        Args:
            reference_signal (np.array): The reference signal.
            test_signal (np.array): The test signal.

        Returns:
            bool: True if pattern is detected, False otherwise.
        """
        ref_length = len(reference_signal)
        for t in range(len(test_signal) - ref_length + 1):
            segment = test_signal[t:t + ref_length]
            if euclidean(reference_signal, segment) < self.threshold:
                return True
        return False

    def fit(self, time_series, labels):
        """
        Train the classifier using the given time series and labels.

        Args:
            time_series (np.array): A (N, T) array where N is the number of samples and T is the length of the time series.
            labels (list): The corresponding labels for the time series.
        """
        self.patterns = []

        for label in set(labels):
            class_indices = [i for i, l in enumerate(labels) if l == label]
            class_signals = time_series[class_indices]

            for i in range(time_series.shape[2]):  # Iterate over features
                aggregated_signal = np.mean(class_signals[:, :, i], axis=0)
                discretized_signal = self.piecewise_constant_model(aggregated_signal, n_max=5)
                self.patterns.append((i, discretized_signal, label))

        X, y = [], []
        for idx, obj in enumerate(time_series):
            features = []
            for i, ref_signal, label in self.patterns:
                pattern_detected = self.detect_pattern(ref_signal, obj[:, i])
                features.append(1 if pattern_detected else 0)
            X.append(features)
            y.append(labels[idx])

        self.clf.fit(X, y)

    def predict(self, time_series):
        """
        Predict the labels for the given time series.

        Args:
            time_series (np.array): A (N, T) array where N is the number of samples and T is the length of the time series.

        Returns:
            list: Predicted labels for the input time series.
        """
        predictions = []
        for obj in time_series:
            features = []
            for i, ref_signal, label in self.patterns:
                pattern_detected = self.detect_pattern(ref_signal, obj[:, i])
                features.append(1 if pattern_detected else 0)
            predictions.append(self.clf.predict([features])[0])

        return predictions
