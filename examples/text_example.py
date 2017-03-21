from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
import pandas as pd

from text_selection import TextDiscoveryTool

# simple_train = ['call you tonight', 'Call me a cab', 'please call me...PLEASE!']
# vect = CountVectorizer()
# vect.fit(simple_train)
# print(vect.get_feature_names())
# simple_train_dtm = vect.transform(simple_train)
# data_frame = pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
#
# example_test = ["please don't call me"]
# simple_test_dtm = vect.transform(example_test)


def sms_predict():
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    sms = pd.read_table(url, header=None, names=['label', 'message'])
    sms['label_num'] = sms.label.map({'ham': 0, 'spam': 1})
    X = sms.message
    y = sms.label_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    vect = CountVectorizer()
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_dtm, y_train)

    # transform testing data (using fitted vocabulary) into a document-term matrix

    y_pred_class = nb.predict(X_test_dtm)
    y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]

    print(accuracy_score(y_test, y_pred_class))
    print(confusion_matrix(y_test, y_pred_class))
    print(roc_auc_score(y_test, y_pred_prob))

    y_pred_class = log_reg.predict(X_test_dtm)
    y_pred_prob = log_reg.predict_proba(X_test_dtm)[:, 1]

    print(accuracy_score(y_test, y_pred_class))
    print(confusion_matrix(y_test, y_pred_class))
    print(roc_auc_score(y_test, y_pred_prob))


def yelp_predict():
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/yelp.csv'
    best_and_worst = pd.read_csv(url)
    best_and_worst = best_and_worst[(best_and_worst.stars == 5) | (best_and_worst.stars == 1)]
    X = best_and_worst.text
    y = best_and_worst.stars
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    vect = CountVectorizer()
    X_train_dtm = vect.fit_transform(X_train)
    log_reg = LogisticRegression()
    log_reg.fit(X_train_dtm, y_train)
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)

    print(accuracy_score(y_test, y_pred_class))
    print('Null Accuracy {}%'.format((y_test.value_counts().head(1) / y_test.shape) * 100))
    print(confusion_matrix(y_test, y_pred_class))
    print(classification_report(y_test, y_pred_class))


def easy_yelp_predict():
    text_predict = TextDiscoveryTool(expensive=False)
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/yelp.csv'
    reviews = pd.read_csv(url)
    reviews = reviews[(reviews.stars == 5) | (reviews.stars == 1)]
    text_predict.x = reviews.text
    text_predict.y = reviews.stars
    text_predict.set_data(rand=1)
    text_predict.use_logistic_reg()
    text_predict.classification_report()
    text_predict.accuracy_score()
    text_predict.confusion_matrix()


def easy_sms_predict():
    text_predict = TextDiscoveryTool(expensive=False)
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    sms = pd.read_table(url, header=None, names=['label', 'message'])
    sms['label_num'] = sms.label.map({'ham': 0, 'spam': 1})
    text_predict.x = sms.message
    text_predict.y = sms.label_num
    text_predict.set_data()
    text_predict.classification_report()
    text_predict.accuracy_score()
    text_predict.confusion_matrix()
    text_predict.roc_auc()
    text_predict.use_logistic_reg()
    text_predict.classification_report()
    text_predict.accuracy_score()
    text_predict.confusion_matrix()
    text_predict.roc_auc()

easy_yelp_predict()
