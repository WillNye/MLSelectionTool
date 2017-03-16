from sklearn.datasets import load_iris

from sklearn_selection import PMDiscoveryTool


def iris_example():
    iris = load_iris()
    x = iris.data
    y = iris.target

    iris_data = PMDiscoveryTool(x, x, y)
    iris_data.kneighbors_classifier_selection(weight_options=['uniform', 'distance'])


iris_example()


