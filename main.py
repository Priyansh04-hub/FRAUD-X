from pathlib import Path
import json
import joblib
import pandas as pd

from src.data_loader import load_raw_data
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_features
from src.model_training import train_all_models


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("FRAUD-X")
    print("Financial Fraud Detection Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading datasets...")

    train, test = load_raw_data()

    print(f"Training data: {train.shape}")
    print(f"Testing data: {test.shape}")

    print("\n[2/5] Engineering features...")

    train, test = engineer_features(
        train,
        test
    )

    print(f"Training features: {train.shape}")
    print(f"Testing features: {test.shape}")

    print("\n[3/5] Preparing machine learning data...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = prepare_features(
        train,
        test
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    print("\n[4/5] Training models...")

    trained_models, results = train_all_models(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("\nModel Results")

    results_table = []

    for model_name, metrics in results.items():
        if "error" in metrics:
            print(f"\n{model_name}: FAILED")
            print(metrics["error"])
            continue

        print(f"\n{model_name}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        results_table.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"]
        })

    results_df = pd.DataFrame(results_table)

    if results_df.empty:
        raise RuntimeError("No model completed successfully.")

    best_model_name = (
        results_df
        .sort_values("F1", ascending=False)
        .iloc[0]["Model"]
    )

    best_model = trained_models[best_model_name]

    print("\n[5/5] Saving pipeline artifacts...")

    joblib.dump(
        best_model,
        MODEL_DIR / "fraud_model.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    with open(
        REPORT_DIR / "pipeline_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "selected_model": best_model_name,
                "training_rows": int(train.shape[0]),
                "testing_rows": int(test.shape[0]),
                "feature_count": int(X_train.shape[1])
            },
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("FRAUD-X PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Selected model: {best_model_name}")
    print(f"Model saved: {MODEL_DIR / 'fraud_model.pkl'}")
    print(f"Scaler saved: {MODEL_DIR / 'scaler.pkl'}")
    print(f"Results saved: {REPORT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()