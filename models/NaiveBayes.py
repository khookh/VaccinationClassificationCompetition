import pandas as pd
import numpy as np
import pickle


class NaiveBayes:
    """
    Class defining the Naive Bayes Classifier attributes and methods
    """

    def __init__(self, key: str = None, training_set: pd.DataFrame = None, load_path: str = None):
        """
        Constructor of the Naive Bayes Classifier
        :param key: (str) name of the class attribute
        :param training_set: (pd.DataFrame) data set containing training data (row = entries, column = attributes)
        :param load_path: (str) path of the Naive Bayes model to load
        """
        if training_set is not None:
            if key not in training_set.columns:
                raise NameError(f'{key} is not a valid key because this attribute does not exist')
            self.key = key
            self.attributes = np.delete(training_set.columns, np.where(training_set.columns == self.key))
            self.prob_map = dict.fromkeys(training_set[key].unique())
            self.prob_key = dict(training_set[key].value_counts(normalize=True))
            self.build_prob(training_set)
        elif load_path is not None:
            self.load(load_path)
        else:
            raise ValueError('You must give in input either a valid training set or a path to a Naive Bayes model (see '
                             'documentation)')

    def build_prob(self, training_set):
        """
        Constructs the conditional probabilities mapping
        :param training_set: (pd.DataFrame) data set containing training data (row = entries, column = attributes)
        """
        for key in self.prob_map.keys():
            self.prob_map[key] = [dict(training_set[training_set[self.key] == key]
                                       [attribute].value_counts(normalize=True)) for attribute in self.attributes]

    def predict(self, input_row):
        """
        :param input_row: Array containing the input vector
        :return: (dict) Dictionary with class attribute's values as keys and prediction probability as value
        """
        return {key: self.prob_key[key] *
                np.prod([self.prob_map[key][i][input_row[i]]
                        for i in range(len(self.attributes))]) for key in self.prob_map.keys()}

    def save(self, path):
        """
        :param path: Location where to store the Naive Bayes model
        """
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, path):
        """
        :param path: Location where is stored a valid Naive Bayes model
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
    test = NaiveBayes(training_set=test_df, key="Buy")

    print(test.predict(np.array(["High", "More", "Big", "High"])))  # YES is the expected value here
    print(test.predict(np.array(["Low", "5", "Medium", "Medium"])))  # NO is the expected value here
    print(test.predict(np.array(["Low", "4", "Medium", "High"])))  # YES is the expected value here

    test.save("test_bayes")

    test_2 = NaiveBayes(load_path="test_bayes")

    print(test_2.predict(np.array(["High", "More", "Big", "High"])))  # YES is the expected value here
    print(test_2.predict(np.array(["Low", "4", "Medium", "Medium"])))  # NO is the expected value here
