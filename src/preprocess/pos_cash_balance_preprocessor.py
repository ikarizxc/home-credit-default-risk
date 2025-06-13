import pandas as pd
from src.preprocess.base_preprocessor import BasePreprocessor


class PosCashBalancePreprocessor(BasePreprocessor):
    """
    Класс для препроцессинга данных POS_CASH_balance.
    """
    def __init__(self, n=None):
        """
        Загружает POS_CASH_balance CSV.

        Args:
            n (Optional[int]): Кол-во загружаемых строк из csv файлов.
        """
        self.POS_CASH_balance = pd.read_csv('data/POS_CASH_balance.csv', nrows=n)
        
    def get_prepared_data(self):
        """
        Препроцесс POS_CASH_balance. Дамми энкодим категориальные фичи и агрегируем все признаки: 'sum', 'min', 'max', 'mean'.
        
        Returns:
            POS_CASH_balance_transformed (pd.DataFrame): Предобработанный POS_CASH_balance.
        """     
        POS_CASH_balance_dummy = self._dummy_encode_categorical_features(self.POS_CASH_balance)
        
        POS_CASH_balance_transformed = pd.concat([
            POS_CASH_balance_dummy.groupby('SK_ID_CURR')[['SK_ID_PREV', 'MONTHS_BALANCE']].count(),
            POS_CASH_balance_dummy.groupby('SK_ID_CURR')[[col for col in POS_CASH_balance_dummy.columns if col not in ['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE']]].agg(['sum', 'min', 'max', 'mean']),
        ], axis=1)

        POS_CASH_balance_transformed.columns = POS_CASH_balance_transformed.columns.map(
            lambda col: 'p_c_b__' + ('_'.join(col) if isinstance(col, tuple) else col)
        )

        POS_CASH_balance_transformed = POS_CASH_balance_transformed.reset_index()
        
        return POS_CASH_balance_transformed