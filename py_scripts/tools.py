# File containing function used for data preparation
import pandas as pd
import numpy as np


def mean_feature_clustered(data_frame: pd.DataFrame, feature_name: str, list_features: list, numeric: bool,
                           source_df: pd.DataFrame):
    """
    It returns a data_frame where the NaN value of the given 'feature_name' are replaced by the
    mean values of the 'feature_name' that match the given filters of list_features
    :param numeric: (bool) if the feature is numeric or not
    :param data_frame: (pd.DataFrame) data frame containing all the dataset
    :param feature_name: (str) name of the feature we want to replace the NaN values
    :param list_features: (list) of the feature we consider for clustering
    :return: updated data_frame
    """

    def filters_mean(row):
        """
        :param row: row where the considered feature has a NaN value to handle
        :return: mean(max) value of the rows matching the row features listed in list_features
        """
        bool_val = [source_df[feature] == row[feature] for feature in list_features]
        for elem in bool_val[1:-1]: bool_val[0] = np.logical_and(bool_val[0], elem)
        filtered_df = source_df[bool_val[0]]
        return filtered_df[feature_name].mean() if numeric else filtered_df[feature_name].value_counts().index[0]

    lnan = data_frame[data_frame[feature_name].isna()]
    lnan[feature_name] = lnan.apply(lambda row: filters_mean(row), axis=1)
    data_frame.loc[lnan.index.tolist(), feature_name] = lnan[feature_name]
    return data_frame


def normalization(data_frame: pd.DataFrame, feature_name: str):
    """
    Normalize the value of the given feature (x-mean / standard_deviation)
    :param data_frame: (pd.DataFrame)
    :param feature_name: eature (column) of the data_frame that need to be normalized
    :return: an updated copy of the column
    """
    c = data_frame[feature_name].copy()
    c_mean = c.mean()
    c_std = c.std()
    return c.apply(lambda x: (x - c_mean) / c_std)


def scaling(data_frame: pd.DataFrame, feature_name: str):
    """
    Scale the values of the given feature between 0 and 1
    :param data_frame: (pd.DataFrame)
    :param feature_name: feature (column) of the data_frame that need to be scaled
    :return: an updated copy of the column
    """
    c = data_frame[feature_name].copy()
    c_min = c.min()
    c_max = c.max()
    return c.apply(lambda x: (x - c_min) / (c_max - c_min))
