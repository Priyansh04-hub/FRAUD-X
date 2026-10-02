import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "unix_time",
    "hour",
    "day_of_week",
    "month",
    "age",
    "merchant_frequency",
    "customer_frequency",
    "category_frequency",
    "category_encoded"
]

def prepare_features(train, test):
    missing_train = [
        feature for feature in FEATURES
        if feature not in train.columns
    ]

    missing_test = [
        feature for feature in FEATURES
        if feature not in test.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing training features: {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Missing testing features: {missing_test}"
        )

    X_train = train[FEATURES].copy()
    X_test = test[FEATURES].copy()

    y_train = train["is_fraud"].copy()
    y_test = test["is_fraud"].copy()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=FEATURES,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=FEATURES,
        index=X_test.index
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )