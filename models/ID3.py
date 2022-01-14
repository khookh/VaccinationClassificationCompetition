import pandas as pd
import numpy as np
import pickle


class ID3:
    """
    Class defining the ID3 tree attributes and methods
    """

    # TODO : implements regression mode
    # TODO : implements a print function that displays the tree's structure
    def __init__(self, key: str = None, training_set: pd.DataFrame = None, p_type: str = "classification",
                 load_path: str = None):
        """
        Constructor of the ID3 tree model
        :param key: (str) name of the class attribute
        :param training_set: (pd.DataFrame) data set containing training data (row = entries, column = attributes)
        :param p_type: (str) type of prediction expected, default = "classification", option = "regression"
        :param load_path: (str) path of the ID3 to load
        """
        self._model = None
        if training_set is not None:
            if p_type not in ["classification", "regression"]:
                raise NameError(f'{p_type} is not accepted as a prediction type for ID3, see documentation')
            if key not in training_set.columns:
                raise NameError(f'{key} is not a valid key because this attribute does not exist')
            self.training_set = training_set
            self.key = key
            self.p_type = p_type
            self.attributes = np.delete(training_set.columns, np.where(training_set.columns == self.key))
            self._key_list = self.training_set[self.key].unique()
            self.build_tree(df=self.training_set, attributes=self.attributes)
        elif load_path is not None:
            self.load(load_path)
        else:
            raise ValueError('You must give in input either a valid training set or a path to an ID3 model (see '
                             'documentation)')

    def build_tree(self, df: pd.DataFrame, attributes: np.array, node=None, option=None):
        """
        This methods built the ID3 tree recursively upon calling
        :param df: (pd.DataFrame) (sub-)DataFram still considered on this step of the tree building
        :param attributes: (np.array) list of attributes still considered on this step of the tree building
        :param node: current node (None for root iteration)
        :param option: option leading to the current node (None for root iteration)
        :return:
        """
        # compute split entropy for each attribute left
        split_etp = {elem: self.split_entropy(df, elem) for elem in attributes}
        # lowest split entropy = greatest information gain -> attribute to be selected
        greatest_IG = min(split_etp, key=split_etp.get)
        # create new node of the ID3 tree
        n_node = self.ID3_node(greatest_IG, df[greatest_IG].unique())

        if node is None:
            # initialize the model on the root node
            self._model = n_node
            node = self._model
        else:
            node.assign(option, n_node)
        for elem in n_node.children.keys():
            resulting_df = df[df[greatest_IG] == elem]
            # if only one output key left at this step of the tree -> stop recursion
            if len(df[self.key].unique()) == 1 or len(attributes) == 1:
                if option is None:
                    # in the case where we stop at the root node -> very rare case, only with small parameters
                    [node.assign(option, df[self.key].iloc[0]) for option in df[greatest_IG].unique()]
                else:
                    node.assign(option, df[self.key].iloc[0])
                break
            else:
                self.build_tree(attributes=np.delete(attributes, np.where(attributes == greatest_IG)), node=n_node,
                                df=resulting_df, option=elem)

    def predict(self, input_row):
        """
        :param input_row: Array containing the input vector
        :return: The prediction made from the ID3 model processing of the input vector
        """
        if self._model is None:
            raise PermissionError(
                'You must first build the tree with ID3.built_tree() before using ID3.predict(input_row)')
        if len(input_row) != len(self.attributes):
            raise ValueError(f'Received {input_row} as input, the number of input features does not match the model : '
                             f'{len(self.attributes)}')
        node = self._model
        try:
            _input = input_row[np.where(self.attributes == node.attribute)[0][0]]
            while not (node.children[_input] in self._key_list):
                node = node.children[_input]
                _input = input_row[np.where(self.attributes == node.attribute)[0][0]]
            return node.children[_input]
        except TypeError as ve:
            print(f'An error occurred while processing the input features with the ID3 model, check the integrity of '
                  f'the input features : {ve}')

    def entropy_key(self, df: pd.DataFrame):
        """
        :param df: (pd.DataFrame)
        :return: The Entropy on key value of the given (sub-)DataFrame
        """
        # If there are k Key values, the entropy is defined as the sum of :
        # (-) the probability of occurrence of each value * by the log2 of the probability
        return sum([-elem * np.log2(elem) for elem in df[self.key].value_counts(normalize=True)])

    def split_entropy(self, df: pd.DataFrame, attribute: str):
        """
        :param df: (pd.DataFrame)
        :param attribute: (str) The feature on which the split entropy must be computed
        :return: The split entropy for the given feature
        """
        # The Split-Entropy is defined as the weighted sum of the entropy calculated for the sub-DataFrames made of
        # grouping of each valid value of the given feature
        val_count = df[attribute].value_counts(normalize=True)
        return sum([self.entropy_key(df[df[attribute] == elem]) * val_count.at[elem] for elem in val_count.keys()])

    def save(self, path):
        """
        :param path: Location where to store the ID3 model
        """
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, path):
        """
        :param path: Location where is stored a valid ID3 model
        """
        f = open(path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    class ID3_node:
        """
        Sub-Class defining ID3 tree's nodes
        """

        def __init__(self, attribute, options=None):
            """
            :param attribute: Attribute (feature) that defines the current split done on this node
            :param options: Options that are still left from this node on this step of the tree building
            """
            self.attribute = attribute
            if options is not None:
                self.children = dict.fromkeys(options)

        def assign(self, option, node):
            """
            :param option: Input option to the current node's attribute
            :param node: Node following the current's node for the given input option
            """
            self.children[option] = node


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
    test = ID3(training_set=test_df, key="Buy")

    print(test.predict(np.array(["High", "More", "Big", "High"])))  # YES is the expected value here
    print(test.predict(np.array(["Low", "5", "Medium", "Medium"])))  # NO is the expected value here

    test.save("test_ID3")

    test_2 = ID3(load_path="test_ID3")
    print(test_2.predict(np.array(["High", "More", "Big", "High"])))  # YES is the expected value here
