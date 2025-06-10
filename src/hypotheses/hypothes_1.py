from src.model_trainer import ModelTrainer
from src.preprocess.application_preprocessor import ApplicationPreprocessor


class HypothesCorrelation(ModelTrainer):
    def __init__(self, model, params_grid=None, test_size=0.3, n=None):
        super().__init__(model, params_grid, test_size, n)
    
    def _prepare_data(self):
        application = ApplicationPreprocessor(self.n)
        application.delete_high_correlation_features()
        return application.get_prepared_data()
