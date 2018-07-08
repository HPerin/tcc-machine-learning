import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


# --------------------- PREPROCESS GLOBAL ---------------------


def _preprocess_quiz_fields(df: pd.DataFrame) -> pd.DataFrame:
    with open('quiz_fields.txt') as file:
        data = file.read().split('\n')
    df = df[data]
    return df


def _preprocess_transform(df: pd.DataFrame) -> pd.DataFrame:
    with open('transform_values.txt') as file:
        data = file.read().split('\n')
    for row in data:
        row = row.split(' ')
        col = row[0]
        old_value = row[1]
        new_value = row[2]
        if new_value == 'NA':
            df[col] = df[col].replace(np.asarray([old_value], df[col].dtype), np.nan)
        else:
            df[col] = df[col].replace(np.asarray([old_value], df[col].dtype), np.asarray([new_value], df[col].dtype))
    return df


def _preprocess_mandatory(df: pd.DataFrame) -> pd.DataFrame:
    with open('mandatory_columns.txt') as file:
        data = file.read().split('\n')
    df = df.dropna(axis=0, thresh=1, subset=data)
    return df


def _preprocess_ignored(df: pd.DataFrame) -> pd.DataFrame:
    with open('ignored_columns.txt') as file:
        data = file.read().split('\n')
    df: pd.DataFrame = df.drop(data, 1)
    return df


def _preprocess_invalid(df: pd.DataFrame) -> pd.DataFrame:
    for col in df:
        column = df[col]
        if column.count() < 0.95 * len(column):
            df = df.drop(col, 1)

    for index, row in df.iterrows():
        if row.count() < 0.95 * len(row):
            df = df.drop(index, 0)
    return df


def _preprocess_equal_class_size(df: pd.DataFrame) -> pd.DataFrame:
    current = None
    for index, row in df.iterrows():
        if current is None:
            current = row['diabet']
        else:
            if row['diabet'] != current:
                current = row['diabet']
            else:
                df = df.drop(index, 0)
    return df


def preprocess_global(df: pd.DataFrame) -> pd.DataFrame:
    df = _preprocess_quiz_fields(df)
    df = _preprocess_transform(df)
    df = _preprocess_ignored(df)
    df = _preprocess_mandatory(df)
    df = _preprocess_equal_class_size(df)
    df = _preprocess_invalid(df)
    return df


# --------------------- PREPROCESS LOCAL ---------------------


def _preprocess_fill(df: pd.DataFrame) -> pd.DataFrame:
    for col in df:
        df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].mean())
        if df[col].isnull().sum() > 0:
            raise Exception('column %s is nan' % col)
    return df


def _preprocess_normalize(df: pd.DataFrame) -> pd.DataFrame:
    for col in df:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def preprocess_local(df: pd.DataFrame) -> pd.DataFrame:
    df = _preprocess_fill(df)
    df = _preprocess_normalize(df)
    return df
