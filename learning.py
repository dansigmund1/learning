from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
import pandas as pd
import numpy as np

# ############################## SupervisedClassifier Model ##############################

class SupervisedClassifier:
    def __init__(self, data):
        self.data = data
    def make_quality_features(df):
        features = pd.DataFrame(index=df.index)
        # fraction of missing values in the row
        features["missing_rate"] = df.isnull().mean(axis=1)
        # number of out‑of‑range values (example: age between 0 and 120)
        if "age" in df.columns:
            features["age_outlier"] = ((df["age"] < 0) | (df["age"] > 120)).astype(int)
        # add more rules as needed
        return features

    X = make_quality_features(df)  # your features
    y = df["is_bad"]               # 0/1 labels you defined

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    def check_data_quality(df, model, feature_func):
        X = feature_func(df)
        scores = model.predict_proba(X)[:, 1]  # probability of being bad
        return scores

    quality_scores = check_data_quality(new_df, model, make_quality_features)
    new_df["quality_score"] = quality_scores

# ############################## UnsupervisedAnomalyDetection Model ##############################

class UnsupervisedAnomalyDetection:
    def __init__(self, data):
        self.data = data

    def make_quality_features(df):
        features = pd.DataFrame(index=df.index)
        # fraction of missing values in the row
        features["missing_rate"] = df.isnull().mean(axis=1)
        # number of out‑of‑range values (example: age between 0 and 120)
        if "age" in df.columns:
            features["age_outlier"] = ((df["age"] < 0) | (df["age"] > 120)).astype(int)
        # add more rules as needed
        return features

    X = make_quality_features(df).dropna()  # numeric features only
    anomaly_model = IsolationForest(contamination=0.05)  # assume ~5% anomalies
    anomaly_model.fit(X)

    df["anomaly_score"] = anomaly_model.predict(X)  # -1 = anomaly, 1 = normal
    df["is_bad"] = (df["anomaly_score"] == -1).astype(int)

    def check_data_quality(df, model, feature_func):
        X = feature_func(df)
        scores = model.predict_proba(X)[:, 1]  # probability of being bad
        return scores

    quality_scores = check_data_quality(new_df, model, make_quality_features)
    new_df["quality_score"] = quality_scores


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Input data", required=True)
    parser.add_argument("-m", "--model", help="Which learning model to use", required=True)
    parser.add_argument("-nc", "--new_check", help="Adds a new data quality check", required=False)
    
    sc = SupervisedClassifier()
    print(sc.check_data_quality)

    uad = UnsupervisedAnomalyDetection()
    print(uad)