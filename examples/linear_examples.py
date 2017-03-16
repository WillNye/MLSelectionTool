import pandas as pd

from sklearn_selection import PMDiscoveryTool

# data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
# feature_cols = ['TV', 'Radio', 'Newspaper']
#
# iris = load_iris()
# x = iris.data
# y = iris.target
#
# iris_data = PMDiscoveryTool(x, x, y)
# iris_data.kneighbors_classifier_selection(weight_options=['uniform', 'distance'])


def pima_example():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv(url, header=None, names=col_names)

    feature_cols = ['pregnant', 'insulin', 'bmi', 'age']

    pima_data = PMDiscoveryTool(pima, pima[feature_cols], pima.label)
    pima_data.set_feature_cols(feature_cols)

    pima_data.logistic_regression_selection()
    pima_data.linear_regression_selection()


pima_example()


