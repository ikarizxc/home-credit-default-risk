from abc import ABC, abstractmethod
from datetime import datetime
from src.model_trainer import ModelTrainer

class BaseHypothesRunner(ABC):
    """
    Базовый класс для гипотез.
    """
    def __init__(self, n=None):
        """
        Args:
            n (Optional[int]): Кол-во загружаемых строк из csv файлов.
        """
        self.model_trainer = None
        self.n = n
      
    @abstractmethod  
    def _get_prepared_data(self):
        """
        Метод для получения готовых данных для обучения модели и получения предсказаний.
        
        Args:
            tuple[pd.DataFrame, pd.Series, pd.DataFrame]: X_train, y_train, X_test.
        """
        pass
    
    def run(self, model, params_grid: dict=None, test_size: float=0.3):
        """
        Метод для обучения модели, создания файла submission.
        
        Args:
            model: Инициализированная модель для обучения.
            params_grid (Optional[dict]): Параметры для GridSearchCV.
            test_size (Optional[float]): Размер валидационной выборки [0-1].
        """
        
        self.model_trainer = ModelTrainer(model, params_grid=params_grid)
        
        X_train, y_train, X_test = self._get_prepared_data()
        
        self.model_trainer.train_model(X_train, y_train, test_size)
        self.model_trainer.get_submission(X_test, f'{self.__class__.__name__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')