import pandas as pd
import numpy as np


class ID3_node:
    def __init__(self, attribute, options=None):
        self.attribute = attribute
        if options is not None:
            self.children = dict.fromkeys(options)

    def assign(self, option, node):
        """
        :param option: input option to the current node's attribute
        :param node: node following the current's node for the given input option
        """
        self.children[option] = node


class ID3:
    """
    Class
    """
    # TODO : implements regression mode
    _model = None

    def __init__(self, training_set: pd.DataFrame, key: str, p_type="classification"):
        """
        Class constructor for the ID3 tree model
        :param training_set: (pd.DataFrame) data set containing training data (row = entries, column = attributes)
        :param key: (str) name of the class attribute
        :param p_type: (str) type of prediction expected, default = "classification", option = "regression"
        """
        self.training_set = training_set
        self.key = key
        self.p_type = p_type
        if p_type not in ["classification", "regression"]:
            raise NameError(f'{p_type} is not accepted as a prediction type for ID3, see documentation')
        if key not in training_set.columns:
            raise NameError(f'{key} is not a valid key because this attribute does not exist')
        self._attributes = training_set.columns
        self._attributes = np.delete(self._attributes, np.where(self._attributes == self.key))

        self._key_list = self.training_set[self.key].unique()

    def build_tree(self, attributes=None, node=None, df=None, option=None):
        if df is None:
            df = self.training_set
        if attributes is None:
            attributes = self._attributes
        # compute split entropy for each attribute left
        split_etp = {elem: self.split_entropy(df, elem) for elem in attributes}
        # lowest split entropy = greatest information gain -> attribute to be selected
        greatest_IG = min(split_etp, key=split_etp.get)
        remaining_attributes = np.delete(attributes, np.where(attributes == greatest_IG))
        # create new node of the ID3 tree
        n_node = ID3_node(greatest_IG, df[greatest_IG].unique())

        if node is None:
            self._model = n_node
        else:
            node.assign(option, n_node)
        for elem in n_node.children.keys():
            resulting_df = df[df[greatest_IG] == elem]
            if len(df[self.key].unique()) == 1:
                node.assign(option, df[self.key].iloc[0])
                break
            else:
                self.build_tree(attributes=np.delete(attributes, np.where(attributes == greatest_IG)), node=n_node,
                                df=resulting_df, option=elem)

    def predict(self, input_row):
        if self._model is None:
            raise PermissionError(
                'You must first build the tree with ID3.built_tree() before using ID3.predict(input_row)')
        node = self._model
        input = input_row[np.where(self._attributes == node.attribute)][0]
        while not (node.children[input] in self._key_list):
            node = node.children[input]
            input = input_row[np.where(self._attributes == node.attribute)][0]
        return node.children[input]

    def entropy_key(self, df: pd.DataFrame):
        """
        :param df: (pd.DataFrame)
        :return: The Entropy on key value of the given (sub-)DataFrame
        """
        # If there are k Key values, the entropy is defined as the sum of :
        # (-) the probability of occurrence of each value * by the log2 of the probability
        return sum([-elem * np.log2(elem) for elem in df[self.key].value_counts(normalize=True)])

    def split_entropy(self, df: pd.DataFrame, attribute):
        val_count = df[attribute].value_counts(normalize=True)
        return sum([self.entropy_key(df[df[attribute] == elem]) * val_count.at[elem] for elem in val_count.keys()])


# For testing purpose
if __name__ == '__main__':
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

    test.build_tree()

    print(test.predict(np.array(["High", "More", "Big", "High"]))) # YES is the expected value here
