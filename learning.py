#!/usr/bin/env python3

import argparse
import json

class Learning:
    def __init__(self):
        self.quality_features = "files/data_quality_features.json"
    
    def make_quality_features(self, df, new_check=None):
        if new_check:
            with open(self.quality_features, 'w') as dqc:
                checks = json.load(dqc)
                checks.get('data_quality_checks',[]).append(new_check)
            json.dump(checks)
        with open(self.quality_features, 'r') as qf:
            quality_features = json.load(qf)
        features = quality_features.get('data_quality_checks',[])
        return features
    
    def get_model(self, features, model):
        if model.lower() in ['supervised classifier','sc']:
            from models.supervised_classifier import SupervisedClassifier
            return 'supervised_classifier'
        elif model.lower() in ['unsupervised anomaly detection','uad']:
            from models.unsupervised_anomaly_detection import UnsupervisedAnomalyDetection
            return 'unsupervised_anomaly_detection'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Input data", required=True)
    parser.add_argument("-m", "--model", help="Which learning model to use", required=True)
    parser.add_argument("-nc", "--new_check", help="Adds a new data quality check", required=False)
    args = parser.parse_args()

    learning = Learning()
    features = learning.make_quality_features(args.new_check)
    model = learning.get_model(features, args.model)