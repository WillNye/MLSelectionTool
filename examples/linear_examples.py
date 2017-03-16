import pandas as pd

from sklearn_selection import PMDiscoveryTool


def pima_example():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv(url, header=None, names=col_names)

    feature_cols = ['pregnant', 'insulin', 'bmi', 'age']

    pima_data = PMDiscoveryTool(pima, pima[feature_cols], pima.label, expensive=False)
    pima_data.set_feature_cols(feature_cols)

    pima_data.logistic_regression_selection()
    pima_data.linear_regression_selection()


pima_example()


