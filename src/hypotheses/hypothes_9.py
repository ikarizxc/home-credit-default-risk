from src.hypotheses.base_hypothes_runner import BaseHypothesRunner
from src.preprocess.application_preprocessor import ApplicationPreprocessor

class HypothesContactsNumber(BaseHypothesRunner):
    def __init__(self, n=None):
        super().__init__(n)
    
    def _get_prepared_data(self):
        application = ApplicationPreprocessor(self.n)
        application.delete_high_correlation_features()
        application.add_days_percents_features()
        application.add_agg_ext_sources()
        application.add_documents_count()
        application.add_credit_features()
        application.add_social_circle_feature()
        application.add_working_hours()
        application.add_bad_car()
        application.add_contacts_number()
        
        X_train, y_train, X_test = application.get_prepared_data()
        return X_train, y_train, X_test