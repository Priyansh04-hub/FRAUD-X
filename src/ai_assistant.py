class FraudInvestigationAssistant:
    def generate_report(
        self,
        transaction,
        risk_result,
        important_features=None
    ):
        risk_level = risk_result["risk_level"]
        fraud_probability = risk_result["fraud_probability"]
        anomaly_score = risk_result["anomaly_score"]
        threshold = risk_result.get("fraud_threshold", 0.10)
        threshold_triggered = risk_result.get(
            "threshold_triggered",
            False
        )
        investigation_triggered = risk_result.get(
            "investigation_triggered",
            False
        )

        amount = transaction.get("amt", "Unknown")
        category = transaction.get("category", "Unknown")
        merchant = transaction.get("merchant", "Unknown")

        report = {
            "risk_level": risk_level,
            "fraud_probability": fraud_probability,
            "anomaly_score": anomaly_score,
            "transaction_amount": amount,
            "merchant": merchant,
            "category": category,
            "findings": [],
            "important_features": important_features or [],
            "recommendation": ""
        }

        if fraud_probability >= threshold:
            report["findings"].append(
                f"Fraud probability exceeded the {threshold:.0%} investigation threshold."
            )
        else:
            report["findings"].append(
                f"Fraud probability remained below the {threshold:.0%} investigation threshold."
            )

        if risk_level == "HIGH":
            report["findings"].append(
                "The combined risk engine classified the transaction as high risk."
            )
        elif risk_level == "MEDIUM":
            report["findings"].append(
                "The combined risk engine classified the transaction as medium risk."
            )
        else:
            report["findings"].append(
                "The combined risk engine classified the transaction as low risk."
            )

        if important_features:
            report["findings"].append(
                "Important model features: "
                + ", ".join(important_features)
            )

        if investigation_triggered:
            report["recommendation"] = (
                "Transaction should receive additional investigation."
            )
        else:
            report["recommendation"] = (
                "Transaction does not require immediate investigation."
            )

        return report

    def format_report(self, report):
        lines = [
            "FRAUD-X INVESTIGATION REPORT",
            f"Risk Level: {report['risk_level']}",
            f"Fraud Probability: {report['fraud_probability']:.2%}",
            f"Anomaly Score: {report['anomaly_score']:.4f}",
            f"Transaction Amount: {report['transaction_amount']}",
            f"Merchant: {report['merchant']}",
            f"Category: {report['category']}",
            "",
            "Findings:"
        ]

        lines.extend(
            f"- {finding}"
            for finding in report["findings"]
        )

        if report.get("important_features"):
            lines.extend([
                "",
                "Important Model Features:"
            ])

            lines.extend(
                f"- {feature}"
                for feature in report["important_features"]
            )

        lines.extend([
            "",
            "Recommendation:",
            report["recommendation"]
        ])

        return "\n".join(lines)