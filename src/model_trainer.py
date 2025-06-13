import os
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score

class ModelTrainer():
    """
    Класс для обучения модели с опциональным GridSearchCV и сохранения submission.
    """
    def __init__(
        self,
        model,
        params_grid: Optional[dict] = None
    ) -> None:
        """
        Инициализирует ModelTrainer.

        Args:
            model: Объект модели sklearn.
            params_grid (Optional[dict]): Словарь параметров для GridSearchCV.
        """
        super().__init__()
        self._grid_search = None
        self._model = None
        self._set_model(model, params_grid)
        self._use_grid_search = params_grid != None
        self._test_ids = []
        
    def _set_model(
        self,
        model,
        model_params: Optional[dict]
    ) -> None:
        """
        Настраивает модель или GridSearchCV.

        Args:
            model: Объект модели sklearn.
            model_params (Optional[dict]): Параметры для GridSearchCV.
        """
        if model_params == None:
            self._model = model
            return
            
        self._grid_search = GridSearchCV(model, model_params, scoring='roc_auc', verbose=1)
        self._model = None
    
    def _train_model(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series
    ) -> None:
        """
        Обучает модель (с grid search или без) и вычисляет ROC-AUC на валидации.

        Args:
            X_train (pd.DataFrame): Признаки для тренировки.
            X_val (pd.DataFrame): Признаки для валидации.
            y_train (pd.Series): Целевой признак для тренировки.
            y_val (pd.Series): Целевой признак для валидации.
        """
        if self._use_grid_search:
            self._grid_search.fit(X_train, y_train)
            self._model = self._grid_search.best_estimator_
        else:
            self._model.fit(X_train, y_train)
            
        if hasattr(self._model, "predict_proba"):
            val_probas = self._model.predict_proba(X_val)[:, 1]
            self._roc_auc_score = roc_auc_score(y_val, val_probas)
        elif hasattr(self._model, "decision_function"):
            val_scores = self._model.decision_function(X_val)
            self._roc_auc_score = roc_auc_score(y_val, val_scores)
        else:
            raise AttributeError("Model has no method to produce scores for ROC-AUC")
            
        print(f"Model fitted")
        print(f"ROC-AUC on validation data = {self._roc_auc_score}")
        if self._use_grid_search:
            print(f"Best model params: {self._grid_search.best_params_}")
    
    def _train_val_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Делит данные на тренировочные и валидационные.

        Args:
            X (pd.DataFrame): Признаки.
            y (pd.Series): Целевой признак.
            test_size (float): Доля валидации.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                X_train, X_val, y_train, y_val.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=True)
        return X_train, X_val, y_train, y_val
    
    def _save_submission(
        self,
        test_ids: pd.Series,
        predicted_probas: np.ndarray,
        file_name: str
    ) -> None:        
        """
        Сохраняет submission в CSV-файл.

        Args:
            test_ids (pd.Series): Идентификаторы тестовых записей.
            predicted_probas (np.ndarray): Предсказанные вероятности.
            file_name (str): Имя файла без расширения.
        """
        submission = pd.DataFrame()

        submission['SK_ID_CURR'] = test_ids
        submission['TARGET'] = predicted_probas
        
        os.makedirs('submissions', exist_ok=True)
        submission.to_csv(f'submissions/{file_name}.csv', index=False)
        print(f"Submission file saved to 'submissions/{file_name}.csv'")
        
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.3,
        drop_id: bool = True
    ) -> None:        
        """
        Полный процесс обучения с разделением данных.

        Args:
            X (pd.DataFrame): Полный набор признаков.
            y (pd.Series): Целевой признак.
            test_size (float): Доля валидационной выборки.
            drop_id (bool): Удалять ли столбец SK_ID_CURR.
        """
        if drop_id:
            X = X.drop(columns='SK_ID_CURR')
        X_train, X_val, y_train, y_val = self._train_val_split(X, y, test_size)
        self._train_model(X_train, X_val, y_train, y_val)
        
    def get_submission(
        self,
        X_test: pd.DataFrame,
        file_name: str,
        drop_id: bool = True
    ) -> None:
        """
        Генерирует и сохраняет CSV с предсказаниями для теста.

        Args:
            X_test (pd.DataFrame): Набор тестовых признаков.
            file_name (str): Имя файла без расширения.
            drop_id (bool): Удалять ли столбец SK_ID_CURR.
        """
        test_ids = X_test['SK_ID_CURR']
        if drop_id:
            X_test = X_test.drop(columns='SK_ID_CURR')
        self._save_submission(test_ids, self._model.predict_proba(X_test)[:, 1], file_name)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Возвращает DataFrame с важностью признаков.

        Returns:
            pd.DataFrame: Столбцы ['feature', 'importances'] отсортированные по убыванию.
        """
        if hasattr(self._model, 'feature_importances_'):
            return pd.DataFrame({
                'importances': self._model.feature_importances_,
                'feature': self._model.feature_names_in_
            }).sort_values(by='importances', ascending=False)
        elif hasattr(self._model, "coef_"):
            return pd.DataFrame({
                'importances': np.abs(self._model.coef_).ravel(),
                'feature': self._model.feature_names_in_
            }).sort_values(by='importances', ascending=False)
        else:
            raise AttributeError("Model you fitted doesn't has attrubite to get importances")