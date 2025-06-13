from src.hypotheses.base_hypothes_runner import BaseHypothesRunner
from src.preprocess.application_preprocessor import ApplicationPreprocessor
    
class HypothesDaysPercents(BaseHypothesRunner):
    """
    Гипотеза о соотношении дней.
    """
    def __init__(self, n=None):
        super().__init__(n)
    
    def _get_prepared_data(self):
        application = ApplicationPreprocessor(self.n)
        application.delete_high_correlation_features()
        application.add_days_percents_features()
        X_train, y_train, X_test = application.get_prepared_data()
        return X_train, y_train, X_test