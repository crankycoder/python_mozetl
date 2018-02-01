from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import exp
import numpy


# Helper function to do some math for cllr.
def neg_log_sig(log_odds):
    neg_log_odds = [-1.0 * x for x in log_odds]
    e = numpy.exp(neg_log_odds)
    return [numpy.log(1 + f) for f in e if f < (f + 1)]


# Compute the log likelihood ratio cost which should be minimized.
def cllr(lrs_on_target, lrs_off_target):
    # based on Niko Brummer's original implementation:
    # Niko Brummer and Johan du Preez, Application-Independent Evaluation of Speaker Detection"
    # Computer Speech and Language, 2005
    lrs_on_target = numpy.log(lrs_on_target[~numpy.isnan(lrs_on_target)])
    lrs_off_target = numpy.log(lrs_off_target[~numpy.isnan(lrs_off_target)])

    c1 = numpy.mean(neg_log_sig(lrs_on_target)) / numpy.log(2)
    c2 = numpy.mean(neg_log_sig(-1.0 * lrs_off_target)) / numpy.log(2)
    return (c1 + c2) / 2


def evalcllr(recommendations_list, unmasked_addons):
    # Organizer function to extract weights from recommendation list for passing to cllr.
    lrs_on_target_helper = [item[1] for item in recommendations_list if item[0] in unmasked_addons]
    lrs_off_target_helper = [item[1] for item in recommendations_list if item[0] not in unmasked_addons]
    return cllr(lrs_on_target_helper, lrs_off_target_helper)


# Load a sample of the current Telemetry data that contains clients with >= 3 addons installed
def load_training_data(sc):
    dataset = list()
    # TODO: query telemetry to get a set of recent clients (filtering by profile creation date)
    # TODO: filter based on number of addons installed (>=3)
    # TODO: remove unnecessary columns to get a feature vector that TAAR will happily accept
    # and provide recommendations for
    return dataset

# All this is not needed, delete after getting a functional version
# Convert string column to float
# def str_column_to_float(dataset, column):
#    for row in dataset:
#        row[column] = float(row[column].strip())


# Convert string column to integer
# def str_column_to_int(dataset, column):
#    class_values = [row[column] for row in dataset]
#    unique = set(class_values)
#    lookup = dict()
#    for i, value in enumerate(unique):
#        lookup[value] = i
#    for row in dataset:
#        row[column] = lookup[row[column]]
#    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    # TODO: inside the cross validation we also need to handle the masking of installed addons for the training set.
    # one way to do this would be to precompute a binary mask for each chunk, or implement this in the
    # "evaluate_algorithm" function.
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for fold_i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
# This will be the cllr code now.
# TODO: include the (prediction, confidence) tuple in the input vector so that evalCLLR can be called.
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    # TODO: here the algorithm argument shoudl be a list of taar modules.
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        # TODO this should just call the appropriate TAAR module (collaborative, similarity, or locale)
        # and get the recommendations over the (masked) training clients.
        # 1- The unmasked installed addons in that client is "actual"
        # 2- the output of the TAAR module that was called is "predicted"
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        # TODO: accuracy will now be CLLR
        scores.append(accuracy)
    return scores

# ALL THIS SHIT NEEDS TO GO:

# # Calculate the Euclidean distance between two vectors
# def euclidean_distance(row1, row2):
#     distance = 0.0
#     for i in range(len(row1) - 1):
#         distance += (row1[i] - row2[i]) ** 2
#     return sqrt(distance)
#
#
# # Locate neighbors for a new row
# def get_neighbors(train, test_row, num_neighbors):
#     distances = list()
#     for train_row in train:
#         dist = euclidean_distance(test_row, train_row)
#         distances.append((train_row, dist))
#     distances.sort(key=lambda tup: tup[1])
#     neighbors = list()
#     for i in range(num_neighbors):
#         neighbors.append(distances[i][0])
#     return neighbors
#
#
# # Make a prediction with kNN
# def knn_predict(model, test_row, num_neighbors=2):
#     neighbors = get_neighbors(model, test_row, num_neighbors)
#     output_values = [row[-1] for row in neighbors]
#     prediction = max(set(output_values), key=output_values.count)
#     return prediction
#
#
# # Prepare the kNN model
# def knn_model(train):
#     return train
#
#
# # Make a prediction with weights
# def perceptron_predict(model, row):
#     activation = model[0]
#     for i in range(len(row) - 1):
#         activation += model[i + 1] * row[i]
#     return 1.0 if activation >= 0.0 else 0.0
#
#
# # Estimate Perceptron weights using stochastic gradient descent
# def perceptron_model(train, l_rate=0.01, n_epoch=5000):
#     weights = [0.0 for i in range(len(train[0]))]
#     for epoch in range(n_epoch):
#         for row in train:
#             prediction = perceptron_predict(weights, row)
#             error = row[-1] - prediction
#             weights[0] = weights[0] + l_rate * error
#             for i in range(len(row) - 1):
#                 weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
#     return weights
#
#
# # Make a prediction with coefficients
# def logistic_regression_predict(model, row):
#     yhat = model[0]
#     for i in range(len(row) - 1):
#         yhat += model[i + 1] * row[i]
#     return 1.0 / (1.0 + exp(-yhat))
#
#
# # Estimate logistic regression coefficients using stochastic gradient descent
# def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
#     coef = [0.0 for i in range(len(train[0]))]
#     for epoch in range(n_epoch):
#         for row in train:
#             yhat = logistic_regression_predict(coef, row)
#             error = row[-1] - yhat
#             coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
#             for i in range(len(row) - 1):
#                 coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
#     return coef


# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
    stacked_row = list()
    for i in range(len(models)):
        prediction = predict_list[i](models[i], row)
        stacked_row.append(prediction)
    stacked_row.append(row[-1])
    return row[0:len(row) - 1] + stacked_row


# Stacked Generalization Algorithm
# TODO: thisis where the actual work is, we need to get a comparable flow with the TAAR models as with these other
# standard ML models, the gradient decent
def stacking(train, test):
    # TODO model list now refers to instantiated TAAR recommender modules.
    model_list = [knn_model, perceptron_model]
    predict_list = [knn_predict, perceptron_predict]
    models = list()
    for i in range(len(model_list)):
        model = model_list[i](train)
        models.append(model)
    stacked_dataset = list()
    for row in train:
        stacked_row = to_stacked_row(models, predict_list, row)
        stacked_dataset.append(stacked_row)
    stacked_model = logistic_regression_model(stacked_dataset)
    predictions = list()
    for row in test:
        stacked_row = to_stacked_row(models, predict_list, row)
        stacked_dataset.append(stacked_row)
        prediction = logistic_regression_predict(stacked_model, stacked_row)
        prediction = round(prediction)
        predictions.append(prediction)
    return predictions


# Test stacking on the sonar dataset
seed(1)
# load and prepare data
dataset = []
# this will be a sample from telemetry.

# TODO: load a sample of clients who have several addons installed, include filtering for
# non-sideloaded and ensure diversity sampling accross the addons space is preserved.

n_folds = 3
scores = evaluate_algorithm(dataset, stacking, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
