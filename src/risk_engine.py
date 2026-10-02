class RiskEngine:
    def __init__(
        self,
        fraud_weight=0.7,
        anomaly_weight=0.3
    ):
        self.fraud_weight = fraud_weight
        self.anomaly_weight = anomaly_weight

    def normalize_anomaly_score(
        self,
        score,
        minimum,
        maximum
    ):
        if maximum == minimum:
            return 0.0

        normalized = (
            (score - minimum) /
            (maximum - minimum)
        )

        return max(
            0.0,
            min(1.0, normalized)
        )

    def calculate_risk(
        self,
        fraud_probability,
        anomaly_score,
        anomaly_min,
        anomaly_max
    ):
        normalized_anomaly = self.normalize_anomaly_score(
            anomaly_score,
            anomaly_min,
            anomaly_max
        )

        risk_score = (
            fraud_probability * self.fraud_weight +
            normalized_anomaly * self.anomaly_weight
        )

        if risk_score >= 0.75:
            risk_level = "HIGH"
        elif risk_score >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "fraud_probability": float(fraud_probability),
            "anomaly_score": float(anomaly_score),
            "normalized_anomaly_score": float(normalized_anomaly),
            "risk_score": float(risk_score),
            "risk_level": risk_level
        }