# File containing function used for data preparation


def filters_mean(row, data_frame, feature_name, list_features):
    bool_val = [data_frame[feature] == row[feature] for feature in list_features]
    bool_ = bool_val[0]
    for i in range(len(list_features) - 1): bool_ = bool_ & bool_val[i + 1]
    filtered_df = data_frame[bool_]
    return filtered_df[feature_name].mean()


def mean_feature_clustered_numeric(data_frame, feature_name, list_features):
    """
    It returns a data_frame where the NaN value of the given 'feature_name' are replaced by the
    mean values of the 'feature_name' that match the given filters of list_features
    :param data_frame: data frame containing all the dataset
    :param feature_name: name of the feature we want to replace the NaN values
    :param list_features: list of the feature we consider for clustering
    :return: updated data_frame
    """
    lnan = data_frame[data_frame[feature_name].isna()]
    lnan[feature_name] = lnan.apply(lambda row: filters_mean(row, data_frame, feature_name, list_features), axis=1)
    data_frame.loc[lnan.respondent_id, feature_name] = lnan[feature_name]
    return data_frame
