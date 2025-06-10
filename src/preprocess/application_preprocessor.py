import numpy as np
import pandas as pd

from src.preprocess.base_preprocessor import BasePreprocessor


class ApplicationPreprocessor(BasePreprocessor):
    def __init__(self, n=None):
        self._application_train = pd.read_csv('data/application_train.csv', nrows=n)
        self._application_test = pd.read_csv('data/application_test.csv', nrows=n)
        self._test_ids = self._application_test['SK_ID_CURR']
        
        self._base_preprocess()
        
    def get_prepared_data(self):
        self._preprocessed = self._preprocess(self._preprocessed)
        
        X = pd.concat([self._other, self._preprocessed], axis=1)
                
        X_train = X[X['is_train'] == 1][self._preprocessed.columns]
        X_test = X[X['is_train'] == 0][self._preprocessed.columns]
        
        return X_train, self._y, X_test, self._test_ids
        
    def add_family_status(self):
        self._preprocessed['SINGLE_FAMILY_STATUS'] = (
            (self._preprocessed['NAME_FAMILY_STATUS'] == 'Widow') |
            (self._preprocessed['NAME_FAMILY_STATUS'] == 'Single / not married')
        ).astype('int')
        
    def add_contacts_number(self):
        self._preprocessed['CONTACTS_NUMBER'] = \
            self._preprocessed['FLAG_MOBIL'] + \
            self._preprocessed['FLAG_WORK_PHONE'] + \
            self._preprocessed['FLAG_PHONE'] + \
            self._preprocessed['FLAG_EMAIL']
        
    def add_bad_car(self):
        self._preprocessed['BAD_CAR'] = (
            (self._preprocessed['FLAG_OWN_CAR'] == 0) |
            (self._preprocessed['OWN_CAR_AGE'] > 10) 
        ).astype('int')
        
    def add_working_hours(self):
        self._preprocessed['IS_HOURS_WORKING'] = (
            self._preprocessed['HOUR_APPR_PROCESS_START']
                .between(8, 18)
                .astype(int)
        )
        
    def add_social_circle_feature(self):
        self._preprocessed['HAS_BAD_PERS_IN_SOC_CIRCLE'] = (self._preprocessed['DEF_30_CNT_SOCIAL_CIRCLE'] > 0).astype('int')
        
    def add_credit_features(self):
        self._preprocessed['CREDIT_INCOME_RATIO'] = self._preprocessed['AMT_CREDIT'] / self._preprocessed['AMT_INCOME_TOTAL']
        self._preprocessed['ANNUITY_CREDIT_RATIO'] = self._preprocessed['AMT_ANNUITY'] / self._preprocessed['AMT_CREDIT']
        self._preprocessed['CREDIT_MONTHS'] = self._preprocessed['AMT_CREDIT'] / self._preprocessed['AMT_ANNUITY']
        self._preprocessed['INITIAL_CREDIT_PAY'] = self._preprocessed['AMT_GOODS_PRICE'] - self._preprocessed['AMT_CREDIT']
        
    def add_documents_count(self):
        self._preprocessed['DOCUMENTS_COUNT'] = self._preprocessed[[col for col in self._preprocessed.columns.values if col.startswith('FLAG_DOCUMENT')]].sum(axis=1)
        
    def add_agg_ext_sources(self):
        self._preprocessed["EXT_SOURCE_MIN"] = self._preprocessed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
        self._preprocessed["EXT_SOURCE_MAX"] = self._preprocessed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
        self._preprocessed["EXT_SOURCE_MEAN"] = self._preprocessed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        self._preprocessed["EXT_SOURCE_STD"] = self._preprocessed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        self._preprocessed["EXT_SOURCE_MIN_MAX_DIV"] = self._preprocessed['EXT_SOURCE_MIN'] / self._preprocessed['EXT_SOURCE_MAX']
        self._preprocessed["EXT_SOURCE_WEIGHTED"] = (self._preprocessed['EXT_SOURCE_1'] + 5 * self._preprocessed['EXT_SOURCE_2'] + 3 * self._preprocessed['EXT_SOURCE_3']) / 3
        
    def add_days_percents_features(self):
        self._preprocessed['DAYS_EMP_BIRTH_PERCENT'] = self._preprocessed['DAYS_EMPLOYED'] / self._preprocessed['DAYS_BIRTH']
        self._preprocessed['DAYS_REG_BIRTH_PERCENT'] = self._preprocessed['DAYS_REGISTRATION'] / self._preprocessed['DAYS_BIRTH']
        self._preprocessed['DAYS_PUB_BIRTH_PERCENT'] = self._preprocessed['DAYS_ID_PUBLISH'] / self._preprocessed['DAYS_BIRTH']
        
    def delete_high_correlation_features(self):
        self._preprocessed = self._delete_high_correlation_features(self._preprocessed)
        
    def _base_preprocess(self):
        X, y = self._concat_train_test()
        
        ignore_features = ['is_train', 'SK_ID_CURR']
        relevant_features = [col for col in X.columns if col not in ignore_features]
        
        self._preprocessed = X[relevant_features]
        self._other = X[ignore_features]
        self._y = y 
    
    def _concat_train_test(self):
        self._application_train['is_train'] = 1
        self._application_test['is_train'] = 0

        y = self._application_train['TARGET']
        self._application_train.drop('TARGET', axis=1, inplace=True)

        data = pd.concat([self._application_train, self._application_test])
        
        return data, y
    
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data.loc[data['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = np.nan
        data = self._delete_duplicates(data)
        data = self._cap_outliers(data)
        data = self._dummy_encode_categorical_features(data)
        return data
