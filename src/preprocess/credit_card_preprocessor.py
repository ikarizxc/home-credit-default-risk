import pandas as pd
from src.preprocess.base_preprocessor import BasePreprocessor


class CreditCardPreprocessor(BasePreprocessor):
    def __init__(self, n=None):
        self.credit_card_balance = pd.read_csv('data/credit_card_balance.csv', nrows=n)
        
    def get_prepared_data(self):
        credit_card_balance_dummy = self._dummy_encode_categorical_features(self.credit_card_balance)
        
        credit_card_balance_transformed = pd.concat([
            credit_card_balance_dummy.groupby('SK_ID_CURR')[['MONTHS_BALANCE', 'SK_ID_PREV']].count(),
            credit_card_balance_dummy.groupby('SK_ID_CURR')[[col for col in credit_card_balance_dummy.columns if col not in ['MONTHS_BALANCE', 'SK_ID_PREV', 'SK_ID_CURR']]].agg(['sum', 'min', 'max', 'mean']),
        ], axis=1)

        credit_card_balance_transformed.columns = credit_card_balance_transformed.columns.map(
            lambda col: 'c_c_b__' + ('_'.join(col) if isinstance(col, tuple) else col)
        )

        credit_card_balance_transformed = credit_card_balance_transformed.reset_index()
        
        return credit_card_balance_transformed