from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import json

class SupervisedClassifier:
    def __init__(self, features):
        self.data_quality_features = features

    def make_quality_features(self, new_check=None):
        if new_check:
            with open(self.quality_features, 'w') as dqc:
                checks = json.load(dqc)
                checks.get('data_quality_checks',[]).append(new_check)
            json.dump(checks)
        with open(self.data_quality_features, 'r') as qf:
            quality_features = json.load(qf)
        features = quality_features.get('data_quality_checks',[])
        return features

    def train_and_evaluate_model(self, df, new_check):
        X = self.make_quality_features(new_check)  # your features
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

    # quality_scores = check_data_quality(new_df, model, make_quality_features)
    # new_df["quality_score"] = quality_scores