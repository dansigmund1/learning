import argparse

class Learning:
    def __init__(self):
        return
    
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
    model = learning.get_model(args.model)