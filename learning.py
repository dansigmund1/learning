#!/usr/bin/env python3

import argparse
import pandas as pd

class Learning:
    def __init__(self, data):
        self.quality_features = "files/data_quality_features.json"
        self.data = data
    
    def read_data_into_df(self):
        df = pd.read_csv(self.data, sep="|")
        df["is_bad"] = 0
        return df
    
    def get_model(self, df, model, new_check):
        if model.lower() in ['supervised classifier','sc']:
            from models.supervised_classifier import SupervisedClassifier
            sc = SupervisedClassifier(self.quality_features)
            model = sc.train_and_evaluate_model(df, new_check)
            scores = sc.check_data_quality(model, self.quality_features)
            return scores
        elif model.lower() in ['unsupervised anomaly detection','uad']:
            from models.unsupervised_anomaly_detection import UnsupervisedAnomalyDetection
            return 'unsupervised_anomaly_detection'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Input data", required=True)
    parser.add_argument("-m", "--model", help="Which learning model to use", required=True)
    parser.add_argument("-nc", "--new_check", help="Adds a new data quality check", required=False)
    args = parser.parse_args()

    learning = Learning(args.data)
    df = learning.read_data_into_df()
    model = learning.get_model(df, args.model, args.new_check)