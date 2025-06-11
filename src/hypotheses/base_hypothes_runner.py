from abc import ABC, abstractmethod
from datetime import datetime
from src.model_trainer import ModelTrainer

class BaseHypothesRunner(ABC):
    def __init__(self, n=None):
        self.model_trainer = None
        self.n = n
      
    @abstractmethod  
    def _get_prepared_data(self):
        pass
    
    def run(self, model, params_grid=None, test_size=0.3):
        self.model_trainer = ModelTrainer(model, params_grid=params_grid)
        
        X_train, y_train, X_test = self._get_prepared_data()
        
        self.model_trainer.train_model(X_train, y_train, test_size)
        self.model_trainer.get_submission(X_test, f'{self.__class__.__name__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')