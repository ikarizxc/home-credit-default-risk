import pandas as pd
from src.preprocess.base_preprocessor import BasePreprocessor


class PreviousApplicationsPreprocessor(BasePreprocessor):
    """
    Класс для препроцессинга данных previous_application.
    """
    def __init__(self, n=None):
        """
        Загружает previous_application CSV.

        Args:
            n (Optional[int]): Кол-во загружаемых строк из csv файлов.
        """
        self.previous_applications = pd.read_csv('data/previous_application.csv')
        
    def get_prepared_data(self):
        """
        Препроцесс previous_application. Дамми энкодим категориальные фичи и агрегируем все признаки: 'sum', 'min', 'max', 'mean'.
        
        Returns:
            previous_application_transformed (pd.DataFrame): Предобработанный previous_application.
        """    
        previous_applications_dummy = self._dummy_encode_categorical_features(self.previous_applications)
        
        previous_application_transformed = pd.concat([
            previous_applications_dummy.groupby('SK_ID_CURR')[['SK_ID_PREV']].count(),
            previous_applications_dummy.groupby('SK_ID_CURR')[[col for col in previous_applications_dummy.columns if col not in ['SK_ID_PREV', 'SK_ID_CURR']]].agg(['sum', 'min', 'max', 'mean']),
        ], axis=1)

        previous_application_transformed.columns = previous_application_transformed.columns.map(
            lambda col: 'p_a__' + ('_'.join(col) if isinstance(col, tuple) else col)
        )

        previous_application_transformed = previous_application_transformed.reset_index()
        
        return previous_application_transformed