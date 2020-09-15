import pandas as pd
import numpy as np
from numpy import log2 as log

data_set = {'Taste': ['Salty', 'Spicy', 'Spicy', 'Spicy', 'Spicy', 'Sweet', 'Salty', 'Sweet', 'Spicy', 'Salty'],
            'Temperature': ['Hot', 'Hot', 'Hot', 'Cold', 'Hot', 'Cold', 'Cold', 'Hot', 'Cold', 'Hot'],
            'Texture': ['Soft', 'Soft', 'Hard', 'Hard', 'Hard', 'Soft', 'Soft', 'Soft', 'Soft', 'Hard'],
            'Eat': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes']}

data_frame = pd.DataFrame(data=data_set)
eps = np.finfo(float).eps


def find_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    return entropy


def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[
        attribute].unique()
    entropy2 = 0
    den = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)


def find_winner(df):
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def build_tree(df, tree=None):
    node = find_winner(df)
    attValue = np.unique(df[node])
    if tree is None:
        tree = {node: {}}

    for value in attValue:

        subtable = get_subtable(df, node, value)
        clValue, counts = np.unique(subtable['Eat'], return_counts=True)

        if len(counts) == 1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = build_tree(subtable)

    return tree


print(build_tree(df=data_frame))

x = {'Taste': {'Salty': {'Texture': {'Hard': 'Yes', 'Soft': 'No'}},
               'Spicy': {'Temperature': {'Cold': {'Texture': {'Hard': 'No', 'Soft': 'Yes'}},
                                         'Hot': {'Texture': {'Hard': 'Yes', 'Soft': 'No'}}}},
               'Sweet': 'Yes'}}
