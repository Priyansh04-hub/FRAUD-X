import numpy as np

class FraudPredictor:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

    def prepare(self, features):
        values = np.asarray(features, dtype=float)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        if self.scaler is not None:
            values = self.scaler.transform(values)

        return values

    def predict(self, features):
        prepared = self.prepare(features)

        prediction = self.model.predict(prepared)[0]

        probability = self.model.predict_proba(
            prepared
        )[0][1]

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability)
        }

    def predict_batch(self, features):
        prepared = self.prepare(features)

        predictions = self.model.predict(prepared)

        probabilities = self.model.predict_proba(
            prepared
        )[:, 1]

        return predictions, probabilities