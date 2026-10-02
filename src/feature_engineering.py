import pandas as pd
import numpy as np

def create_time_features(df):
    data = df.copy()

    data["trans_date_trans_time"] = pd.to_datetime(
        data["trans_date_trans_time"]
    )

    data["hour"] = data["trans_date_trans_time"].dt.hour
    data["day_of_week"] = data["trans_date_trans_time"].dt.dayofweek
    data["month"] = data["trans_date_trans_time"].dt.month
    data["day"] = data["trans_date_trans_time"].dt.day

    return data

def create_age_feature(df):
    data = df.copy()

    data["dob"] = pd.to_datetime(data["dob"])

    data["age"] = (
        data["trans_date_trans_time"].dt.year -
        data["dob"].dt.year
    )

    return data

def create_frequency_features(train, test):
    train_data = train.copy()
    test_data = test.copy()

    merchant_frequency = train_data["merchant"].value_counts()
    customer_frequency = train_data["cc_num"].value_counts()
    category_frequency = train_data["category"].value_counts()

    train_data["merchant_frequency"] = (
        train_data["merchant"].map(merchant_frequency).fillna(0)
    )

    test_data["merchant_frequency"] = (
        test_data["merchant"].map(merchant_frequency).fillna(0)
    )

    train_data["customer_frequency"] = (
        train_data["cc_num"].map(customer_frequency).fillna(0)
    )

    test_data["customer_frequency"] = (
        test_data["cc_num"].map(customer_frequency).fillna(0)
    )

    train_data["category_frequency"] = (
        train_data["category"].map(category_frequency).fillna(0)
    )

    test_data["category_frequency"] = (
        test_data["category"].map(category_frequency).fillna(0)
    )

    return train_data, test_data

def encode_category(train, test):
    train_data = train.copy()
    test_data = test.copy()

    categories = {
        category: index
        for index, category in enumerate(
            train_data["category"].unique()
        )
    }

    train_data["category_encoded"] = (
        train_data["category"].map(categories).fillna(-1)
    )

    test_data["category_encoded"] = (
        test_data["category"].map(categories).fillna(-1)
    )

    return train_data, test_data

def engineer_features(train, test):
    train_data = create_time_features(train)
    test_data = create_time_features(test)

    train_data = create_age_feature(train_data)
    test_data = create_age_feature(test_data)

    train_data, test_data = create_frequency_features(
        train_data,
        test_data
    )

    train_data, test_data = encode_category(
        train_data,
        test_data
    )

    return train_data, test_data