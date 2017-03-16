import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from utils.linear_model_utils import calc_sens_and_spec, display_results


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
        col_results = calc_sens_and_spec(log_reg_cm, total_predictions)
        print('Initial Threshold ')
        display_results(col_results)

        if not self.expensive:
            # perform a binary search to find the optimal threshold
            threshold = (col_results[len(col_results)-1]['predict_percent']-2) / 100
            most_accurate = {'results': col_results, 'threshold': .5}
            least_deviation = {'results': col_results, 'threshold': threshold}

            for x in range(5):
                y_pred_prob = log_reg.predict_proba(x_test)[:, 1]
                binarize_pred = binarize(y_pred_prob, threshold)[0]
                log_reg_cm = confusion_matrix(y_test, binarize_pred)
                col_results = calc_sens_and_spec(log_reg_cm, total_predictions)
                threshold += .01

                if sum([col['correct_prediction'] for col in col_results]) > sum(
                        [col['correct_prediction'] for col in most_accurate['results']]):
                    most_accurate['results'] = col_results
                    most_accurate['threshold'] = threshold

                if max([c['accuracy'] for c in col_results]) - min(
                        [c['accuracy'] for c in col_results]) < max(
                        [c['accuracy'] for c in least_deviation['results']]) - min(
                        [c['accuracy'] for c in least_deviation['results']]):
                    least_deviation['results'] = col_results
                    least_deviation['threshold'] = threshold

            print('\nThreshold {0:.2f} had the most correct predictions'.format(most_accurate['threshold']))
            display_results(most_accurate['results'])
            print('\nThreshold {0:.2f} had the least deviation predictions'.format(least_deviation['threshold']))
            display_results(least_deviation['results'])

    def linear_regression_selection(self):
        lm = LinearRegression()
        scores = cross_val_score(lm, self.x, self.y, cv=10, scoring='neg_mean_squared_error')
        scores = -scores
        rmse_scores = np.sqrt(scores)
        print('\nLinear Regression Accuracy - {}%'.format(int((1 - rmse_scores.mean()) * 100)))

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

        print('KNeighbors -\n  Accuracy {}%\n  Best Params {},\n  Classifier {}'.format(grid.best_score_ * 100,
                                                                                  grid.best_params_,
                                                                                  grid.best_estimator_))
