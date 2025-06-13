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
        
        self._base_preprocess()
        
    def get_prepared_data(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Применяет пайплайн препроцессинга и возвращает данные.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
                X_train, y_train, X_test.
        """
        self._preprocessed = self._preprocess(self._preprocessed)
        
        X = pd.concat([self._other, self._preprocessed], axis=1)
                
        X_train = X[X['is_train'] == 1].drop(columns=['is_train'])
        X_test = X[X['is_train'] == 0].drop(columns=['is_train'])
        
        return X_train, self._y, X_test
        
    def add_family_status(self) -> None:
        """
        Добавляет бинарный признак SINGLE_FAMILY_STATUS
        (вдова или не замужем - 1).
        """
        self._preprocessed['SINGLE_FAMILY_STATUS'] = (
            (self._preprocessed['NAME_FAMILY_STATUS'] == 'Widow') |
            (self._preprocessed['NAME_FAMILY_STATUS'] == 'Single / not married')
        ).astype('int')
        
    def add_contacts_number(self) -> None:
        """
        Добавляет признак суммарного числа контактов.
        """
        self._preprocessed['CONTACTS_NUMBER'] = \
            self._preprocessed['FLAG_MOBIL'] + \
            self._preprocessed['FLAG_WORK_PHONE'] + \
            self._preprocessed['FLAG_PHONE'] + \
            self._preprocessed['FLAG_EMAIL']
        
    def add_bad_car(self) -> None:
        """
        Добавляет признак: клиент без машины или со старой (>10 лет).
        """
        self._preprocessed['BAD_CAR'] = (
            (self._preprocessed['FLAG_OWN_CAR'] == 0) |
            (self._preprocessed['OWN_CAR_AGE'] > 10) 
        ).astype('int')
        
    def add_working_hours(self) -> None:
        """
        Добавляет признак: заявка была совершена в рабочие часы (8–18) или нет.
        """
        self._preprocessed['IS_HOURS_WORKING'] = (
            self._preprocessed['HOUR_APPR_PROCESS_START']
                .between(8, 18)
                .astype(int)
        )
        
    def add_social_circle_feature(self) -> None:
        """
        Добавляет признак: наличие связей с дефолтом (>0).
        """
        self._preprocessed['HAS_BAD_PERS_IN_SOC_CIRCLE'] = (self._preprocessed['DEF_30_CNT_SOCIAL_CIRCLE'] > 0).astype('int')
        
    def add_credit_features(self) -> None:
        """
        Добавляет признаки: соотношений кредитных величин.
        """
        self._preprocessed['CREDIT_INCOME_RATIO'] = self._preprocessed['AMT_CREDIT'] / self._preprocessed['AMT_INCOME_TOTAL']
        self._preprocessed['ANNUITY_CREDIT_RATIO'] = self._preprocessed['AMT_ANNUITY'] / self._preprocessed['AMT_CREDIT']
        self._preprocessed['CREDIT_MONTHS'] = self._preprocessed['AMT_CREDIT'] / self._preprocessed['AMT_ANNUITY']
        self._preprocessed['INITIAL_CREDIT_PAY'] = self._preprocessed['AMT_GOODS_PRICE'] - self._preprocessed['AMT_CREDIT']
        
    def add_documents_count(self) -> None:
        """
        Добавляет признак: количество поданных документов FLAG_DOCUMENT_*.
        """
        self._preprocessed['DOCUMENTS_COUNT'] = self._preprocessed[[col for col in self._preprocessed.columns.values if col.startswith('FLAG_DOCUMENT')]].sum(axis=1)
        
    def add_agg_ext_sources(self) -> None:
        """
        Добавляет признаки: агрегация EXT_SOURCE_{1,2,3}: min/max/mean/std/ratio/weighted.
        """
        self._preprocessed["EXT_SOURCE_MIN"] = self._preprocessed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
        self._preprocessed["EXT_SOURCE_MAX"] = self._preprocessed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
        self._preprocessed["EXT_SOURCE_MEAN"] = self._preprocessed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        self._preprocessed["EXT_SOURCE_STD"] = self._preprocessed[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        self._preprocessed["EXT_SOURCE_MIN_MAX_DIV"] = self._preprocessed['EXT_SOURCE_MIN'] / self._preprocessed['EXT_SOURCE_MAX']
        self._preprocessed["EXT_SOURCE_WEIGHTED"] = (self._preprocessed['EXT_SOURCE_1'] + 5 * self._preprocessed['EXT_SOURCE_2'] + 3 * self._preprocessed['EXT_SOURCE_3']) / 3
        
    def add_days_percents_features(self) -> None:
        """
        Добавляет признаки соотношения дней: Employment/Birth, Registration/Birth, Publish/Birth.
        """
        self._preprocessed['DAYS_EMP_BIRTH_PERCENT'] = self._preprocessed['DAYS_EMPLOYED'] / self._preprocessed['DAYS_BIRTH']
        self._preprocessed['DAYS_REG_BIRTH_PERCENT'] = self._preprocessed['DAYS_REGISTRATION'] / self._preprocessed['DAYS_BIRTH']
        self._preprocessed['DAYS_PUB_BIRTH_PERCENT'] = self._preprocessed['DAYS_ID_PUBLISH'] / self._preprocessed['DAYS_BIRTH']
        
    def delete_high_correlation_features(self, threshold: float=0.85) -> None:
        """
        Удаляет колонки с высокой корреляцией.
        
        Args:
            threshold (Optional[float]): Порог корреляции
        """
        self._preprocessed = self._delete_high_correlation_features(self._preprocessed, threshold)
        
    def _base_preprocess(self) -> None:
        """
        Объединяет train/test. Делит выборку на выборку для дальнейшего препроцесса и то что не нужно препроцессить
        """
        X, y = self._concat_train_test()
        
        ignore_features = ['is_train', 'SK_ID_CURR']
        relevant_features = [col for col in X.columns if col not in ignore_features]
        
        self._preprocessed = X[relevant_features]
        self._other = X[ignore_features]
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
    
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет последовательный вызов методов препроцессинга: удаление дубликатов, ограничение выбросов, дамми кодирование категориальных признаков.

        Args:
            data (pd.DataFrame): Объединённый DataFrame.

        Returns:
            pd.DataFrame: Обработанные данные.
        """
        data.loc[data['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = np.nan
        data = self._delete_duplicates(data)
        data = self._cap_outliers(data)
        data = self._dummy_encode_categorical_features(data)
        return data
