# MLSelectionTool

Trying to figure out the optimal setup to train your data and predict it with the highest quality can be filled with tons of boiler plate.
That's where the MLSelectionTool comes in.

Simply define the data, the x axis, and the y axis (and feature column if necessary) then compare your data accross various tools with minimal effort.

Example of how easy it is to compare data:

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age']

pima_data = PMDiscoveryTool(pima, pima, pima.label, expensive=False)
print("KNeighbors")
pima_data.kneighbors_classifier_selection()

# Set feature columns on x to perform compare against linear models
pima_data.set_feature_cols(feature_cols)
print("\nLogistic Regression")
pima_data.logistic_regression_selection()
print("\nLinear Regression")
pima_data.linear_regression_selection()

And in the terminal you'd see something similar to:

KNeighbors
0.756510416667
{'n_neighbors': 17}
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=17, p=2,
           weights='uniform')

Logistic Regression
Col 0 - (Accuracy: 90%, Predicted: 85%)
Col 1 - (Accuracy: 24%, Predicted: 14%)

Linear Regression
0.44016290844
