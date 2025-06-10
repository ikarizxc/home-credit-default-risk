import pandas as pd
from src.preprocess.base_preprocessor import BasePreprocessor


class BureauPreprocessor(BasePreprocessor):
    def __init__(self, n=None):
        self.bureau_balance = pd.read_csv('data/bureau_balance.csv', nrows=n)
        self.bureau = pd.read_csv('data/bureau.csv', nrows=n)
        
    def _get_prepared_bureau_balance(self) -> pd.DataFrame:
        bureau_balance_dummy = self._dummy_encode_categorical_features(self.bureau_balance)
        
        bureau_balance_transformed = pd.concat([
            bureau_balance_dummy.groupby('SK_ID_BUREAU')[['MONTHS_BALANCE']].count().rename({'MONTHS_BALANCE': "MONTHS_COUNT"}, axis=1),
            bureau_balance_dummy.groupby('SK_ID_BUREAU')[[col for col in bureau_balance_dummy.columns if col not in ['MONTHS_BALANCE', 'SK_ID_BUREAU']]].agg(['sum', 'min', 'max', 'mean'])
        ], axis=1)
        bureau_balance_transformed.columns = bureau_balance_transformed.columns.map(
            lambda col: 'b_b__' + ('_'.join(col) if isinstance(col, tuple) else col)
        )
        bureau_balance_transformed = bureau_balance_transformed.reset_index()
        
        return bureau_balance_transformed
        
    def get_prepared_data(self):
        bureau_balance_transformed = self._get_prepared_bureau_balance()
        
        bureau_dummy = self._dummy_encode_categorical_features(self.bureau)
        
        bureau_with_balance = pd.merge(bureau_dummy, bureau_balance_transformed, how='left', on='SK_ID_BUREAU')

        bureau_transformed = pd.concat([
            bureau_with_balance.groupby('SK_ID_CURR')[['SK_ID_BUREAU']].count(),
            bureau_with_balance.groupby('SK_ID_CURR')[[col for col in bureau_with_balance.columns if col not in ['SK_ID_CURR', 'SK_ID_BUREAU']]].agg(['sum', 'min', 'max', 'mean'])
        ], axis=1)
        bureau_transformed.columns = bureau_transformed.columns.map(
            lambda col: 'b__' + ('_'.join(col) if isinstance(col, tuple) else col)
        )
        bureau_transformed = bureau_transformed.reset_index()
        
        return bureau_transformed