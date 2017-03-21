import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

from utils.linear_model_utils import calc_sens_and_spec, display_results


class UnsetValueException(Exception):
    pass


class TextDiscoveryTool:
    def __init__(self, expensive=True):
        self.x = None
        self.y = None
        self.x_train, self.x_test, self.y_train, self.y_test = (None, None, None, None)
        self.log_reg, self.vect, self.X_train_dtm = (None, None, None)
        self.expensive = expensive

    def set_data(self, rand=0):
        if self.x is None or self.y is None:
            raise UnsetValueException('Set both x and y before attempting to fit and transform')

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, random_state=rand)
        self.vect = CountVectorizer()
        self.X_train_dtm = self.vect.fit_transform(self.x_train)

        self.log_reg = LogisticRegression()
        self.log_reg.fit(self.X_train_dtm, self.y_train)
        self.X_test_dtm = self.vect.transform(self.x_test)

        self.nb = MultinomialNB()
        self.nb.fit(self.X_train_dtm, self.y_train)
        self.y_pred_class = self.nb.predict(self.X_test_dtm)
        self.y_pred_prob = self.nb.predict_proba(self.X_test_dtm)[:, 1]

    def use_logistic_reg(self):
        self.y_pred_class = self.log_reg.predict(self.X_test_dtm)
        self.y_pred_prob = self.log_reg.predict_proba(self.X_test_dtm)[:, 1]

    def use_naive_bayes(self):
        self.y_pred_class = self.nb.predict(self.X_test_dtm)
        self.y_pred_prob = self.nb.predict_proba(self.X_test_dtm)[:, 1]

    def accuracy_score(self):
        print('\nAccuracy {}%'.format((accuracy_score(self.y_test, self.y_pred_class)) * 100))
        print('Null Accuracy {}%'.format((self.y_test.value_counts().head(1) / self.y_test.shape) * 100))

    def roc_auc(self):
        print('\nRoc Auc Score:')
        print('{}%'.format((roc_auc_score(self.y_test, self.y_pred_prob) * 100)))

    def confusion_matrix(self):
        log_reg_cm = confusion_matrix(self.y_test, self.y_pred_class)
        total_predictions = sum([sum(sub_list) for sub_list in log_reg_cm])
        col_results = calc_sens_and_spec(log_reg_cm, total_predictions)
        print('\nConfusion Matrix: ')
        display_results(col_results)

    def classification_report(self):
        cl_rep = classification_report(self.y_test, self.y_pred_class)
        print('\nClassification Report:')
        print(cl_rep)


