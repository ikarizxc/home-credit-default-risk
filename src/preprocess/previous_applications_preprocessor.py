import pandas as pd
from src.preprocess.base_preprocessor import BasePreprocessor


class PreviousApplicationsPreprocessor(BasePreprocessor):
    def __init__(self, n=None):
        self.previous_applications = pd.read_csv('data/previous_application.csv')
        
    def get_prepared_data(self):
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