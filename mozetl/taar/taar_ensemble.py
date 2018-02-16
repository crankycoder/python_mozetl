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


RECOMMENDERS = [CollaborativeRecommender(), LegacyRecommender(),
              LocaleRecommender(), SimilarityRecommender(),
              RecommendationManager()]

# Calculate accuracy percentage - substitute accuracy percentage for cllr
# We could use 1.0-cllr as a metric to reuse even more existing code.
# This will be the cllr code now.
# TODO: include the (prediction, confidence) tuple in the INPUT vector
# so that eval_cllr can be called
# TODO: still include the input vector "actual" which will be passed
# to the eval_cllr function as unmasked_addons.
def accuracy_metric(actual, predicted):
    """
    Actual and predicted are both dictionaries that map client_id to a
    list of addons
    """
    correct = 0
    total_addons = 0
    for client_id, actual_addons in actual.items():
        total_addons += len(actual_addons)
        predicted_addons = predicted[client_id]
        for predicted_addon, predicted_weight in predicted_addons:
            # TODO: the predicted_weight is going to be lost here - we
            # need to wire it into eval_cllr???
            if predicted_addon in actual_addons:
                correct += 1
    return correct / float(total_addons) * 100.0


class RecommenderProxy:
    def __init__(self, base_recommender):
        self._recommender = base_recommender

    def aggregate_recommend(self, train_set, ignored_test_set, **kwargs):
        """
        This function runs recommendations for a given recommender
        against all client records in the train_set.  
        
        TODO: The testset argument is ignored for now.  Not sure if
        this is correct.

        The single required paramater in kwargs is 'MIN_ADDONSETSIZE'

        Return a dictionary of client_id->recommendations
        """
        results = {}
        for client_data in train_set:
            addon_mask_id = client_data['addon_mask_id']
            recommendations = self._recommender.recommend(client_data, 
                                                          limit=kwargs['MIN_ADDONSETSIZE'])
            results[addon_mask_id] = recommendations
        return results



# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, n_folds, **kwargs):

    MIN_ADDONSETSIZE = kwargs['MIN_ADDONSETSIZE']

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

        # This calls each of the TAAR recommender modules :
        # (collaborative, similarity, or locale)
        # and get the recommendations over the (masked) training clients.
        # 1- The unmasked installed addons in that client is "actual"
        # 2- the output of the TAAR module that was called is "predicted"

        masked_train_set, masked_train_set_removed_addons = xvalidator.mask_addons(train_set)

        #masked_test_set  = xvalidator.mask_addons(test_set)
        masked_test_set  = None
        for recommender in RECOMMENDERS:

            # Note that the masked_test_set is set to None because it
            # is currently ignored
            proxy = RecommenderProxy(recommender)
            predicted = proxy.aggregate_recommend(train_set,
                                                  masked_test_set,
                                                  **kwargs)

            actual = masked_train_set_removed_addons
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
    predict_list = [knn_predict, perceptron_predict]
    models = list()
    for i in range(len(RECOMMENDERS)):
        model = RECOMMENDERS[i](train)
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
# load a sample of clients who have several addons installed, include filtering for
# non-sideloaded and ensure diversity sampling accross the addons space is preserved.

dataset = []
with open('datasample.txt','r') as file_in:
    line = file_in.readline()
    dataset.append(json.loads(line))

# This will be a sample from telemetry.

n_folds = 3
scores = evaluate_algorithm(dataset, n_folds, MIN_ADDONSETSIZE=2)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
