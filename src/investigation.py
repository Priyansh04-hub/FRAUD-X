from pathlib import Path
import joblib
import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import FEATURES
from src.risk_engine import RiskEngine
from src.ai_assistant import FraudInvestigationAssistant


class FraudInvestigator:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent

        self.model = joblib.load(
            base_dir / "models" / "fraud_model.pkl"
        )

        self.scaler = joblib.load(
            base_dir / "models" / "scaler.pkl"
        )

        self.anomaly_detector = joblib.load(
            base_dir / "models" / "isolation_forest.pkl"
        )

        self.train_raw = pd.read_csv(
            base_dir / "data" / "raw" / "fraudTrain.csv"
        )

        train_features, _ = engineer_features(
            self.train_raw,
            self.train_raw.copy()
        )

        self.train_features = train_features

        X_train = train_features[FEATURES].copy()

        X_train_scaled = self.scaler.transform(
            X_train
        )

        train_anomaly_scores = (
            -self.anomaly_detector.score_samples(
                X_train_scaled
            )
        )

        self.anomaly_min = float(
            train_anomaly_scores.min()
        )

        self.anomaly_max = float(
            train_anomaly_scores.max()
        )

        self.risk_engine = RiskEngine()

        self.assistant = FraudInvestigationAssistant()

        self.fraud_threshold = 0.10

        self.feature_importance = (
            self.load_feature_importance(base_dir)
        )

    def load_feature_importance(self, base_dir):
        path = (
            base_dir
            / "reports"
            / "feature_importance.csv"
        )

        if not path.exists():
            return []

        importance = pd.read_csv(path)

        if "feature" not in importance.columns:
            return []

        return (
            importance["feature"]
            .head(5)
            .tolist()
        )

    def prepare_transaction(self, transaction):
        transaction_df = pd.DataFrame(
            [transaction]
        )

        _, processed_transaction = engineer_features(
            self.train_raw,
            transaction_df
        )

        X = processed_transaction[
            FEATURES
        ].copy()

        X_scaled = self.scaler.transform(X)

        return (
            X_scaled,
            processed_transaction
        )

    def prepare_transaction_batch(self, transactions):
        transaction_df = transactions.copy()

        _, processed_transactions = engineer_features(
            self.train_raw,
            transaction_df
        )

        X = processed_transactions[
            FEATURES
        ].copy()

        X_scaled = self.scaler.transform(X)

        return (
            X_scaled,
            processed_transactions
        )

    def calculate_batch_risk(self, transactions):
        X_scaled, processed_transactions = (
            self.prepare_transaction_batch(
                transactions
            )
        )

        fraud_probabilities = (
            self.model.predict_proba(
                X_scaled
            )[:, 1]
        )

        anomaly_scores = (
            -self.anomaly_detector.score_samples(
                X_scaled
            )
        )

        normalized_anomaly = (
            anomaly_scores
            - self.anomaly_min
        ) / (
            self.anomaly_max
            - self.anomaly_min
        )

        normalized_anomaly = normalized_anomaly.clip(
            0,
            1
        )

        risk_scores = (
            fraud_probabilities
            * self.risk_engine.fraud_weight
            + normalized_anomaly
            * self.risk_engine.anomaly_weight
        )

        risk_levels = pd.cut(
            risk_scores,
            bins=[
                -float("inf"),
                0.40,
                0.75,
                float("inf")
            ],
            labels=[
                "LOW",
                "MEDIUM",
                "HIGH"
            ]
        )

        result = pd.DataFrame({
            "fraud_probability": fraud_probabilities,
            "anomaly_score": anomaly_scores,
            "normalized_anomaly_score": normalized_anomaly,
            "risk_score": risk_scores,
            "risk_level": risk_levels.astype(str)
        })

        return result

    def investigate(self, transaction):
        X_scaled, processed_transaction = (
            self.prepare_transaction(
                transaction
            )
        )

        fraud_probability = float(
            self.model.predict_proba(
                X_scaled
            )[0][1]
        )

        model_prediction = int(
            self.model.predict(
                X_scaled
            )[0]
        )

        anomaly_prediction = int(
            self.anomaly_detector.predict(
                X_scaled
            )[0]
        )

        anomaly_score = float(
            -self.anomaly_detector.score_samples(
                X_scaled
            )[0]
        )

        risk_result = (
            self.risk_engine.calculate_risk(
                fraud_probability,
                anomaly_score,
                self.anomaly_min,
                self.anomaly_max
            )
        )

        threshold_triggered = (
            fraud_probability
            >= self.fraud_threshold
        )

        investigation_triggered = (
            threshold_triggered
            or anomaly_prediction == 1
            or risk_result["risk_level"] == "HIGH"
        )

        risk_result["model_prediction"] = (
            model_prediction
        )

        risk_result["anomaly_prediction"] = (
            anomaly_prediction
        )

        risk_result["fraud_threshold"] = (
            self.fraud_threshold
        )

        risk_result["threshold_triggered"] = (
            threshold_triggered
        )

        risk_result["investigation_triggered"] = (
            investigation_triggered
        )

        transaction_data = (
            transaction.to_dict()
            if hasattr(
                transaction,
                "to_dict"
            )
            else dict(transaction)
        )

        report = (
            self.assistant.generate_report(
                transaction_data,
                risk_result,
                self.feature_importance
            )
        )

        return {
            "transaction": transaction_data,
            "processed_transaction": (
                processed_transaction
                .iloc[0]
                .to_dict()
            ),
            "risk": risk_result,
            "report": report
        }

    def format_investigation(
        self,
        investigation
    ):
        return self.assistant.format_report(
            investigation["report"]
        )