import pandas as pd
from sklearn.inspection import permutation_importance

def get_permutation_importance(
    model,
    X,
    y,
    feature_names,
    n_repeats=5,
    random_state=42
):
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1",
        n_jobs=-1
    )

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    return importance.reset_index(drop=True)

def top_features(
    model,
    X,
    y,
    feature_names,
    n=5
):
    importance = get_permutation_importance(
        model,
        X,
        y,
        feature_names
    )

    return importance.head(n)