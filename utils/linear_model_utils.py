

def calc_sens_and_spec(lin_mod_list, total_predictions):
    col_results = list()
    for feature in range(len(lin_mod_list)):
        feature_selection_cnt = sum(lin_mod_list[feature])
        predict_percent = int((feature_selection_cnt / total_predictions) * 100)
        correct_prediction = lin_mod_list[feature][feature]
        accuracy = int((correct_prediction / feature_selection_cnt) * 100)
        col_stats = dict(col_pos=feature,
                         accuracy=accuracy,
                         usage=feature_selection_cnt,
                         predict_percent=predict_percent,
                         correct_prediction=correct_prediction)
        col_results.append(col_stats)
    return col_results


def display_results(results):
    for column in results:
        print('Col {} - '
              'Accuracy: {}%, Predicted: {}%, # of Predictions: {}, # of Uses: {}'.format(column['col_pos'],
                                                                                          column['accuracy'],
                                                                                          column['predict_percent'],
                                                                                          column['correct_prediction'],
                                                                                          column['usage']))

