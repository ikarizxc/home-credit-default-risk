from src.hypotheses.base_hypothes_runner import BaseHypothesRunner
from src.preprocess.application_preprocessor import ApplicationPreprocessor

class HypothesCorrelation(BaseHypothesRunner):
    def __init__(self, n=None):
        super().__init__(n)
    
    def _get_prepared_data(self):
        application = ApplicationPreprocessor(self.n)
        application.delete_high_correlation_features()
        X_train, y_train, X_test = application.get_prepared_data()
        return X_train, y_train, X_test