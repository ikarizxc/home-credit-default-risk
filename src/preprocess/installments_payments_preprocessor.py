import pandas as pd
from src.preprocess.base_preprocessor import BasePreprocessor


class InstallmentsPaymentsPreprocessor(BasePreprocessor):
    def __init__(self, n=None):
        self.installments_payments = pd.read_csv('data/installments_payments.csv', nrows=n)
        
    def get_prepared_data(self):        
        installments_payments_transformed = pd.concat([
            self.installments_payments.groupby('SK_ID_CURR')[['SK_ID_PREV']].count(),
            self.installments_payments.groupby('SK_ID_CURR')[[col for col in self.installments_payments.columns if col not in ['SK_ID_PREV', 'SK_ID_CURR']]].agg(['sum', 'min', 'max', 'mean']),
        ], axis=1)

        installments_payments_transformed.columns = installments_payments_transformed.columns.map(
            lambda col: 'i_p__' + ('_'.join(col) if isinstance(col, tuple) else col)
        )

        installments_payments_transformed = installments_payments_transformed.reset_index()
        
        return installments_payments_transformed