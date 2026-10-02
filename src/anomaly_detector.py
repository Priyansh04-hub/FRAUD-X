import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(
        self,
        n_estimators=150,
        contamination="auto",
        random_state=42
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def predict(self, X):
        predictions = self.model.predict(X)

        return np.where(
            predictions == -1,
            1,
            0
        )

    def score(self, X):
        return -self.model.score_samples(X)

    def predict_with_score(self, X):
        predictions = self.predict(X)
        scores = self.score(X)

        return predictions, scores