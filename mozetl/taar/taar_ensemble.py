# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

import boto3  # noqa
import json        # noqa
import itertools        # noqa
import copy
import numpy        # noqa
from botocore.exceptions import ClientError        # noqa
from random import seed        # noqa
from random import randrange        # noqa
from .xvalidator import XValidator        # noqa

from taar.recommenders import CollaborativeRecommender
from taar.recommenders import LegacyRecommender
from taar.recommenders import LocaleRecommender
from taar.recommenders import RecommendationManager
from taar.recommenders import SimilarityRecommender

import logging

logger = logging.getLogger(__name__)


# Calculate accuracy percentage - substitute accuracy percentage for cllr
# We could use 1.0-cllr as a metric to reuse even more existing code.
# This will be the cllr code now.
# TODO: include the (prediction, confidence) tuple in the INPUT vector
# so that eval_cllr can be called
# TODO: still include the input vector "actual" which will be passed
# to the eval_cllr function as unmasked_addons.
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    # TODO: here the algorithm argument should be a list of taar modules.

    # TODO: copy MIN_ADDONSETSIZE from *args
    MIN_ADDONSETSIZE = 2
    xvalidator = XValidator(n_folds, MIN_ADDONSETSIZE)
    folds = xvalidator.cross_validation_split(dataset)

    scores = list()
    for fold in folds:

        # Create a training set which is the entire list of folds,
        # less the current fold
        train_set = list(folds)
        train_set.remove(fold)

        # train_set needs to be converted from a list of lists into a
        # single monolithic list
        train_set = itertools.chain.from_iterable(train_set)
        test_set = copy.copy(fold)

        # TODO this should just call the appropriate TAAR module (collaborative, similarity, or locale)
        # and get the recommendations over the (masked) training clients.
        # 1- The unmasked installed addons in that client is "actual"
        # 2- the output of the TAAR module that was called is "predicted"
        predicted = algorithm(train_set, test_set, *args)

        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        # TODO: accuracy will now be CLLR
        scores.append(accuracy)
    return scores


# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
    stacked_row = list()
    for model in models:
        prediction = predict_list[i](model, row)
        stacked_row.append(prediction)
    stacked_row.append(row[-1])
    return row[:len(row) - 1] + stacked_row


# Stacked Generalization Algorithm
# TODO: this is where the actual work is, we need to get a comparable flow
# utilizing the TAAR models as with these on the standard ML models,
# the gradient decent algorithm should iterate to a set of meta-parameters
# weighting the recommneder modules.
def stacking(train, test):
    # model list now refers to instantiated TAAR recommender modules.
    model_list = [CollaborativeRecommender(), LegacyRecommender(),
                  LocaleRecommender(), SimilarityRecommender(),
                  RecommendationManager()]


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


# Estimate logistic regression coefficients using stochastic gradient descent
def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            yhat = logistic_regression_predict(coef, row)
            error = row[-1] - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef
-

# Test stacking on the sonar dataset
seed(1)
# load and prepare data returned by load_training_data(sc)
# TODO: load a sample of clients who have several addons installed, include filtering for
# non-sideloaded and ensure diversity sampling accross the addons space is preserved.

dataset = []
# This will be a sample from telemetry.

n_folds = 3
scores = evaluate_algorithm(dataset, stacking, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
