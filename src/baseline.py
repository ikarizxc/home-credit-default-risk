from src.preprocess.application_preprocessor import ApplicationPreprocessor
from .model_trainer import ModelTrainer
    
class Baseline(ModelTrainer):
    def __init__(self, model, params_grid=None, test_size=0.3, n=None):
        super().__init__(model, params_grid, test_size, n)
    
    def _prepare_data(self):
        application = ApplicationPreprocessor(self.n)
        return application.get_prepared_data()