from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=100,
            random_state=42
        )
    }

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

def train_all_models(X_train, y_train, X_test, y_test):
    models = get_models()
    trained_models = {}
    results = {}

    for name, model in models.items():
        try:
            trained_model = train_model(
                model,
                X_train,
                y_train
            )

            metrics = evaluate_model(
                trained_model,
                X_test,
                y_test
            )

            trained_models[name] = trained_model
            results[name] = metrics

        except Exception as error:
            results[name] = {
                "error": str(error)
            }

    return trained_models, results