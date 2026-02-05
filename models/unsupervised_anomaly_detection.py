from sklearn.ensemble import IsolationForest
import pandas as pd

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