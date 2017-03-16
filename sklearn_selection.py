import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class PMDiscoveryTool:
    def __init__(self, init_data, init_x, init_y, expensive=True):
        self.data = init_data
        self.x = init_x
        self.y = init_y
        self.expensive = expensive
        self.feature_cols = None

    def set_feature_cols(self, feature_cols):
        self.feature_cols = feature_cols
        self.x = self.data[feature_cols]

    def logistic_regression_selection(self):
        print("\nLogistic Regression Accuracy")
        log_reg = LogisticRegression()
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, random_state=0)
        log_reg.fit(x_train, y_train)
        y_pred_class = log_reg.predict(x_test)
        log_reg_cm = confusion_matrix(y_test, y_pred_class)
        total_predictions = sum([sum(sub_list) for sub_list in log_reg_cm])
        for feature in range(len(log_reg_cm)):
            feature_selection_cnt = sum([sub_list[feature] for sub_list in log_reg_cm])
            predict_percent = int((feature_selection_cnt/total_predictions) * 100)
            accuracy = int((log_reg_cm[feature][feature]/sum(log_reg_cm[feature])) * 100)
            print('Col {} - (Accuracy: {}%, Predicted: {}%)'.format(feature, accuracy, predict_percent))

    def linear_regression_selection(self):
        lm = LinearRegression()
        scores = cross_val_score(lm, self.x, self.y, cv=10, scoring='neg_mean_squared_error')
        scores = -scores
        rmse_scores = np.sqrt(scores)
        print('\nLinear Regression Accuracy: {}%'.format(int((1 - rmse_scores.mean()) * 100)))

    def kneighbors_classifier_selection(self, min_range=1, max_range=31, weight_options=[]):
        k_range = range(min_range, max_range)
        knn = KNeighborsClassifier(n_neighbors=5)
        grid_dict = dict(n_neighbors=k_range)

        if weight_options:
            grid_dict['weights'] = weight_options

        if self.expensive:
            grid = RandomizedSearchCV(knn, grid_dict, cv=10, scoring='accuracy', n_iter=10, random_state=5, n_jobs=-1)
        else:
            grid = GridSearchCV(knn, grid_dict, cv=10, scoring='accuracy', n_jobs=-1)

        grid.fit(self.x, self.y)

        print(grid.best_score_)
        print(grid.best_params_)
        print(grid.best_estimator_)
