from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np

class BasePreprocessor(ABC):
    """
    Базовый класс для препроцессинга данных.
    Содержит общие шаги: удаление дубликатов, заполнение пропусков, кодирование категорий и т.д.
    """
    @abstractmethod
    def get_prepared_data(self) -> pd.DataFrame:
        """
        Выполняет полный пайплайн препроцессинга и возвращает готовый DataFrame.

        Returns:
            pd.DataFrame: Обработанные данные.
        """
        pass
    
    def _dummy_encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Кодирует категориальные признаки в дамми-переменные.

        Args:
            df (pd.DataFrame): Входной DataFrame.

        Returns:
            pd.DataFrame: DataFrame с замененными категориальными признаками на дамми.
        """
        cat_cols = self._get_categorical_features(df)
        df_dummy_encoded = df.copy()
        for col in cat_cols:
            ser = df_dummy_encoded[col].fillna("MISSING")
            dummies = (
                pd.get_dummies(ser, prefix=col, drop_first=True)
                .astype(int)
            )
            df_dummy_encoded = pd.concat([df_dummy_encoded.drop(col, axis=1), dummies], axis=1)
        return df_dummy_encoded
    
    def _delete_high_correlation_features(self, df: pd.DataFrame, threshold: float=0.85) -> pd.DataFrame:
        """
        Удаляет признаки с корреляцией выше threshold.

        Args:
            df (pd.DataFrame): Входной DataFrame.
            threshold (Optional[float]): Порог корреляции.            

        Returns:
            pd.DataFrame: DataFrame с ограниченными выбросами.
        """
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        corr_matrix = df[num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        mean_corr = upper.mean()
        to_drop = mean_corr[mean_corr > threshold].index.tolist()
        
        df_reduced = df.copy()
        df_reduced.drop(columns=to_drop, inplace=True)
        return df_reduced
    
    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ограничивает выбросы числовых признаков в пределах 1.5 * IQR.

        Args:
            df (pd.DataFrame): Входной DataFrame.

        Returns:
            pd.DataFrame: DataFrame с ограниченными выбросами.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        df_capped = df.copy()
        for col in num_cols:
            ser = df_capped[col].astype(float)
            q1, q3 = ser.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df_capped[col] = ser.clip(lower=low, upper=high)
            
        return df_capped
        
    def _delete_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет дублирующиеся строки из DataFrame.

        Args:
            df (pd.DataFrame): Входной DataFrame.

        Returns:
            pd.DataFrame: DataFrame без дубликатов.
        """
        df_no_duplicates = df.copy()
        return df_no_duplicates.drop_duplicates()
    
    def _fill_null_values(self, df: pd.DataFrame, columns, value) -> pd.DataFrame:
        """
        Заполняет проспуски в данных значением value.

        Args:
            df (pd.DataFrame): Входной DataFrame.
            value: Значение, на которые заменяются пропуски.

        Returns:
            df (pd.DataFrame): DataFrame без пропусков.
        """
        df_filled = df.copy()[columns].fillna(value)
        return df_filled
    
    def _get_categorical_features(self, df: pd.DataFrame) -> List[str]:
        """
        Возвращает список категориальных признаков в DataFrame.

        Args:
            df (pd.DataFrame): Входной DataFrame.

        Returns:
            List[str]: Список имен категориальных столбцов.
        """
        return df.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()