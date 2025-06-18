import numpy as np
import pandas as pd

from src.preprocess.base_preprocessor import BasePreprocessor


class ApplicationPreprocessor(BasePreprocessor):
    """
    Класс для препроцессинга данных application_[train|test].
    """
    def __init__(self, n=None):
        """
        Загружает train и test CSV, объединяет их для единого пайплайна.

        Args:
            n (Optional[int]): Кол-во загружаемых строк из csv файлов.
        """
        self._application_train = pd.read_csv('data/application_train.csv', nrows=n)
        self._application_test = pd.read_csv('data/application_test.csv', nrows=n)
        self._test_ids = self._application_test['SK_ID_CURR']
        
        self._ignore_features = []
        
        self._set_data()
        
    def merge_data(self, data: pd.DataFrame, on: str='SK_ID_CURR', how: str='left'):
        """
        Мёрджит данные по ключу в application данные.
        
        Args:
            data (DataFrame): Данные, которы смёрджатся.
            on (string): Колонка по которой мерджится.
            how (string): {'left', 'right', 'outer', 'inner', 'cross'} Тип мерджа.
        """
        self._X.merge(data, on=on, how=how)
        
    def get_prepared_data(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Применяет пайплайн препроцессинга и возвращает данные.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
                X_train, y_train, X_test.
        """
        self._X.loc[self._X['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = np.nan
        self._X = self._delete_duplicates(self._X)
        self._X[self._get_features_to_preprocess()] = self._cap_outliers(self._X[self._get_features_to_preprocess()])
        
        subset = self._X[self._get_features_to_preprocess()]
        processed = self._dummy_encode_categorical_features(subset)
        self._X.drop(columns=subset.columns, inplace=True)
        self._X = pd.concat([self._X, processed], axis=1)
        
        X_train = self._X[self._X['is_train'] == 1].drop(columns=['is_train'])
        X_test = self._X[self._X['is_train'] == 0].drop(columns=['is_train'])
        
        return X_train, self._y, X_test
        
    def add_family_status(self) -> None:
        """
        Добавляет бинарный признак SINGLE_FAMILY_STATUS
        (вдова или не замужем - 1).
        """
        self._X['SINGLE_FAMILY_STATUS'] = (
            (self._X['NAME_FAMILY_STATUS'] == 'Widow') |
            (self._X['NAME_FAMILY_STATUS'] == 'Single / not married')
        ).astype('int')
        
    def add_contacts_number(self) -> None:
        """
        Добавляет признак суммарного числа контактов.
        """
        self._X['CONTACTS_NUMBER'] = \
            self._X['FLAG_MOBIL'] + \
            self._X['FLAG_WORK_PHONE'] + \
            self._X['FLAG_PHONE'] + \
            self._X['FLAG_EMAIL']
        
    def add_bad_car(self) -> None:
        """
        Добавляет признак: клиент без машины или со старой (>10 лет).
        """
        self._X['BAD_CAR'] = (
            (self._X['FLAG_OWN_CAR'] == 0) |
            (self._X['OWN_CAR_AGE'] > 10) 
        ).astype('int')
        
    def add_working_hours(self) -> None:
        """
        Добавляет признак: заявка была совершена в рабочие часы (8–18) или нет.
        """
        self._X['IS_HOURS_WORKING'] = (
            self._X['HOUR_APPR_PROCESS_START']
                .between(8, 18)
                .astype(int)
        )
        
    def add_social_circle_feature(self) -> None:
        """
        Добавляет признак: наличие связей с дефолтом (>0).
        """
        self._X['HAS_BAD_PERS_IN_SOC_CIRCLE'] = (self._X['DEF_30_CNT_SOCIAL_CIRCLE'] > 0).astype('int')
        
    def add_credit_features(self) -> None:
        """
        Добавляет признаки: соотношений кредитных величин.
        """
        new_features = {
            'CREDIT_INCOME_RATIO': self._X['AMT_CREDIT'] / self._X['AMT_INCOME_TOTAL'],
            'ANNUITY_CREDIT_RATIO': self._X['AMT_ANNUITY'] / self._X['AMT_CREDIT'],
            'CREDIT_MONTHS': self._X['AMT_CREDIT'] / self._X['AMT_ANNUITY'],
            'INITIAL_CREDIT_PAY': self._X['AMT_GOODS_PRICE'] - self._X['AMT_CREDIT'],
        }
        self._X = pd.concat([self._X, pd.DataFrame(new_features, index=self._X.index)], axis=1)
        
    def add_documents_count(self) -> None:
        """
        Добавляет признак: количество поданных документов FLAG_DOCUMENT_*.
        """
        new_features = {
            'DOCUMENTS_COUNT': self._X[[col for col in self._X.columns.values if col.startswith('FLAG_DOCUMENT')]].sum(axis=1),
        }
        self._X = pd.concat([self._X, pd.DataFrame(new_features, index=self._X.index)], axis=1)
        
    def add_agg_ext_sources(self) -> None:
        """
        Добавляет признаки: агрегация EXT_SOURCE_{1,2,3}: min/max/mean/std/ratio/weighted.
        """        
        new_features = {
            "EXT_SOURCE_MIN": self._X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1),
            "EXT_SOURCE_MAX": self._X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1),
            "EXT_SOURCE_MEAN": self._X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1),
            "EXT_SOURCE_STD": self._X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1),
            "EXT_SOURCE_MIN_MAX_DIV": 
                self._X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
                / self._X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1),
            "EXT_SOURCE_WEIGHTED": 
                (
                    self._X['EXT_SOURCE_1'] + 
                    5 * self._X['EXT_SOURCE_2'] + 
                    3 * self._X['EXT_SOURCE_3']
                 ) / 3
        }
        self._X = pd.concat([self._X, pd.DataFrame(new_features, index=self._X.index)], axis=1)
        
    def add_days_percents_features(self) -> None:
        """
        Добавляет признаки соотношения дней: Employment/Birth, Registration/Birth, Publish/Birth.
        """
        new_features = {
            'DAYS_EMP_BIRTH_PERCENT': self._X['DAYS_EMPLOYED'] / self._X['DAYS_BIRTH'],
            'DAYS_REG_BIRTH_PERCENT': self._X['DAYS_REGISTRATION'] / self._X['DAYS_BIRTH'],
            'DAYS_PUB_BIRTH_PERCENT': self._X['DAYS_ID_PUBLISH'] / self._X['DAYS_BIRTH'],
        }
        self._X = pd.concat([self._X, pd.DataFrame(new_features, index=self._X.index)], axis=1)
        
    def delete_high_correlation_features(self, threshold: float=0.85) -> None:
        """
        Удаляет колонки с высокой корреляцией.
        
        Args:
            threshold (Optional[float]): Порог корреляции
        """
        self._X[self._get_features_to_preprocess()] = self._delete_high_correlation_features(self._X[self._get_features_to_preprocess()], threshold)
        
    def _get_features_to_preprocess(self):
        return [col for col in self._X.columns.values if col not in self._ignore_features]
        
    def _set_data(self) -> None:
        """
        Объединяет train/test. Делит выборку на выборку для дальнейшего препроцесса и то что не нужно препроцессить
        """
        X, y = self._concat_train_test()
        
        self._ignore_features = ['is_train', 'SK_ID_CURR']
        
        self._X = X
        self._y = y 
    
    def _concat_train_test(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Добавляет метку is_train и конкатенирует наборы.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: объединённый DataFrame и Series целевого признака.
        """
        self._application_train['is_train'] = 1
        self._application_test['is_train'] = 0

        y = self._application_train['TARGET']
        self._application_train.drop('TARGET', axis=1, inplace=True)

        data = pd.concat([self._application_train, self._application_test])
        
        return data, y
