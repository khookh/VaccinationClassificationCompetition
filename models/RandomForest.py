import pandas as pd
import numpy as np
import pickle

import ID3


class RandomForest:
    """
    Class defining the Random Forest attributes and methods
    """

    # TODO : implements regression mode
    def __init__(self, key: str = None, training_set: pd.DataFrame = None, p_type: str = "classification",
                 load_path: str = None, n_tree: int = 10, n_feat_tree: int = 2, n_item: int = 2):
        """
        Constructor of the Random Forest model
        :param key: (str) name of the class attribute
        :param training_set: (pd.DataFrame) data set containing training data (row = entries, column = attributes)
        :param p_type: (str) type of prediction expected, default = "classification", option = "regression"
        :param load_path: (str) path of the Random Forest model to load
        :param n_tree: (int) Number of tree included in the Random Forest
        :param n_feat_tree: (int) Number of features to include in each tree of the Random Forest
        """
        self._model = None
        if training_set is not None:
            if p_type not in ["classification", "regression"]:
                raise NameError(f'{p_type} is not accepted as a prediction type for Random Forest, see documentation')
            if key not in training_set.columns:
                raise NameError(f'{key} is not a valid key because this attribute does not exist')
            if n_feat_tree > len(training_set.columns) or n_feat_tree < 1:
                raise ValueError(f'{n_feat_tree} is not a valid value for the number of features to include per tree, '
                                 f'see the documentation')
            if n_item > len(training_set) or n_item < 2:
                raise ValueError(f'{n_item} is not a valid value for the number of random samples to take from the '
                                 f'data set, see the documentation')
            if n_tree < 2:
                raise ValueError(f'{n_tree} is not a valid value for the number of tree to include in the forest, '
                                 f'see the documentation')
            self.n_item = n_item
            self.n_feat_tree = n_feat_tree
            self.n_tree = n_tree
            self.training_set = training_set
            self.key = key
            self.p_type = p_type
            self._attributes = np.delete(training_set.columns, np.where(training_set.columns == self.key))
            self.build_forest()
        elif load_path is not None:
            self.load(load_path)
        else:
            raise ValueError('You must give in input either a valid training set or a path to an ID3 model (see '
                             'documentation)')


    def bootstrap(self):
        """
        :return: (pd.DataFrame) Bootstrapped training set
        """
        # The bootstrapping consists in selecting a random set of samples from the original data sets, with replacement
        # And only keep a certain number of randomly selected features
        bootstrap = self.training_set.iloc[
            np.random.choice(np.arange(0, len(self.training_set)), size=self.n_item)
        ][np.append(np.random.choice(self._attributes, size=self.n_feat_tree, replace=False), self.key)]
        # While there are columns with not all inputs represented : start again
        while True in [len(bootstrap[column].unique()) != len(self.training_set[column].unique()) for column in bootstrap.columns]:
            bootstrap = self.training_set.iloc[
                np.random.choice(np.arange(0, len(self.training_set)), size=self.n_item)
            ][np.append(np.random.choice(self._attributes, size=self.n_feat_tree, replace=False), self.key)]
        return bootstrap

    def build_forest(self):
        """
        Build the Random Forest following the principle of Bagging (Bootstrap Aggregation of random Trees)
        """
        self._model = [ID3.ID3(training_set=self.bootstrap(), key=self.key) for i in range(self.n_tree)]

    def predict(self, input_row):
        """
        :param input_row: Array containing the input vector
        :return: The prediction made from the Random Forest model processing of the input vector
        """

        val, count = np.unique(
            [elem.predict([input_row[index][0] for index in [np.where(self._attributes.values == attr)
                                                             for attr in elem.attributes]]) for elem in
             self._model], return_counts=True)
        return val[np.argmax(count)]

    def save(self, path):
        """
        :param path: Location where to store the Random Forest model
        """
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, path):
        """
        :param path: Location where is stored a valid Random Forest model
        """
        f = open(path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)


# For testing purpose
if __name__ == '__main__':
    # Normally it's a good practice to implement unit testing in a separate file with libraries such as unittest

    test_data = [["High", "More", "Medium", "High", "No"], ["High", "More", "Medium", "Medium", "No"],
                 ["Medium", "More", "Medium", "High", "Yes"], ["Low", "5", "Medium", "High", "Yes"],
                 ["Low", "4", "Big", "High", "Yes"], ["Low", "4", "Big", "Medium", "No"],
                 ["Medium", "4", "Big", "Medium", "Yes"], ["High", "5", "Medium", "High", "No"],
                 ["High", "4", "Big", "High", "Yes"], ["Low", "5", "Big", "High", "Yes"],
                 ["High", "5", "Big", "Medium", "Yes"], ["Medium", "5", "Medium", "Medium", "Yes"],
                 ["Medium", "More", "Big", "High", "Yes"], ["Low", "5", "Medium", "Medium", "No"]]
    test_df = pd.DataFrame(data=test_data, columns=["Maintenance", "Persons", "LuggageBoot", "Safety", "Buy"])

    print(test_df)
    test = RandomForest(training_set=test_df, key="Buy", n_item=8, n_tree=40)

    print(test.predict(np.array(["High", "More", "Big", "High"])))  # YES is the expected value here
    print(test.predict(np.array(["Low", "5", "Medium", "Medium"])))  # NO is the expected value here

    test.save("test_RF")

    test_2 = RandomForest(load_path="test_RF")
    print(test_2.predict(np.array(["High", "More", "Big", "High"])))  # YES is the expected value here
