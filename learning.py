import argparse
import pandas as pd

class Learning:
    def __init__(self):
        return
    
    def make_quality_features(df):
        features = pd.DataFrame(index=df.index)
        # fraction of missing values in the row
        features["missing_rate"] = df.isnull().mean(axis=1)
        # number of out‑of‑range values (example: age between 0 and 120)
        if "age" in df.columns:
            features["age_outlier"] = ((df["age"] < 0) | (df["age"] > 120)).astype(int)
        # add more rules as needed
        return features
    
    def get_model(self, model):
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
    model = learning.get_model(args.model)