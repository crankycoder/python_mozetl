# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

# noqa: ignore=E127,E502,E999

import boto3  # noqa
import json        # noqa
import itertools        # noqa
from math import exp
import numpy        # noqa
from random import seed        # noqa
from random import randrange

from taar.recommenders import CollaborativeRecommender
from taar.recommenders import LegacyRecommender
from taar.recommenders import LocaleRecommender
from taar.recommenders import SimilarityRecommender

import logging

log = logging.getLogger(__name__)


RECOMMENDERS = [CollaborativeRecommender(), LegacyRecommender(),
                LocaleRecommender(), SimilarityRecommender()]


def row_to_taar_json(row):
    rDict = row.asDict()
    taar_jdata = {"geo_city": rDict.get('geo_city', ''),
                  "subsession_length": rDict.get('subsession_length', 0),
                  "locale": rDict.get('locale', ''),
                  "os": rDict.get('os', ''),
                  "installed_addons": rDict.get('addon_ids', []),
                  "disabled_addons_ids": [],
                  "bookmark_count": rDict.get('bookmark_count', 0),
                  "tab_open_count": rDict.get('tab_open_count', 0),
                  "total_uri": rDict.get('total_uri', 0),
                  "unique_tlds": rDict.get('unique_tlds', 0)}
    return taar_jdata


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, n_folds, **kwargs):

    MIN_ADDONSETSIZE = kwargs['MIN_ADDONSETSIZE']

    xvalidator = XValidator(n_folds, MIN_ADDONSETSIZE)
    folds = xvalidator.cross_validation_split(dataset)

    # Scores is a matrix with rows of folds, columns of
    # recommenders. Logistic regression will then optimize the matrix
    # to be co-efficient parameters.

    scores = []
    for fold in folds:
        # Create a training set which is the entire list of folds,
        # less the current fold
        train_set = list(folds)
        train_set.remove(fold)

        train_set = (row_to_taar_json(x)
                     for x in itertools.chain.from_iterable(train_set))
        masked_train_set, addon_mask_id_map = xvalidator.mask_addons(train_set)

        # Run each of the recommenders against the training set
        for client_json in masked_train_set:

            # Collect recommendation output for each recommender for
            # this client blob
            recommendation_row = []
            for recommender in RECOMMENDERS:
                if not recommender.can_recommend(client_json):
                    continue

                # TODO: limit recommendations should be parameterized
                # instead of a hardcoded 10
                recommendations = recommender.recommend(client_json,
                                                        10,
                                                        extra_data={})

                actual = addon_mask_id_map[client_json['addon_mask_id']]
                accuracy = eval_cllr(recommendations,
                                     actual)

                recommendation_row.append(accuracy)
            scores.append(recommendation_row)
    return scores


# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(recommendation_outputs, row):
    stacked_row = list()
    for prediction, recommender in zip(recommendation_outputs, RECOMMENDERS):
        prediction = recommender(prediction, row)
        stacked_row.append(prediction)
    stacked_row.append(row[-1])
    return row[0:len(row) - 1] + stacked_row


# Stacked Generalization Algorithm
# this is where the actual work is, we need to get a comparable flow
# utilizing the TAAR models as with these on the standard ML models,
# the gradient decent algorithm should iterate to a set of meta-parameters
# weighting the recommneder modules.
def stacking(train, ignored):
    """
    to_stacked_row: generate a matrix of 4 columns, 1 per
    recommender maintaining order with rows of recommendations.
    Row # = len of train which is the dataset of client_id json
    blobs.
    """

    stacked_dataset = []
    for client_data in train:
        stacked_row = []
        for recommender in RECOMMENDERS:
            # Strip out the weight, just keep the GUID
            stacked_row.append([x[0] for x in recommender.recommend(client_data, 10)])
        stacked_dataset.append(stacked_row)

    predictions = logistic_regression_model(stacked_dataset)
    return predictions


# Make a prediction with coefficients
def logistic_regression_predict(model, row):
    yhat = model[0]
    for i in range(len(row) - 1):
        yhat += model[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))


def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
    """Estimate logistic regression coefficients using stochastic gradient descent
    """
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            yhat = logistic_regression_predict(coef, row)
            error = row[-1] - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef


""" TAAR Ensemble math code starts """


def neg_log_sig(log_odds):
    neg_log_odds = [-1.0 * x for x in log_odds]
    e = numpy.exp(neg_log_odds)
    return [numpy.log(1 + f) for f in e if f < (f + 1)]


def cllr(lrs_on_target, lrs_off_target):
    """Compute the log likelihood ratio cost which should be minimized.
    based on Niko Brummer's original implementation:
    Niko Brummer and Johan du Preez, Application-Independent Evaluation of Speaker Detection"
    Computer Speech and Language, 2005
    """
    lrs_on_target = numpy.log(lrs_on_target[~numpy.isnan(lrs_on_target)])
    lrs_off_target = numpy.log(lrs_off_target[~numpy.isnan(lrs_off_target)])

    c1 = numpy.mean(neg_log_sig(lrs_on_target)) / numpy.log(2)
    c2 = numpy.mean(neg_log_sig(-1.0 * lrs_off_target)) / numpy.log(2)
    return (c1 + c2) / 2


def eval_cllr(recommendations_list, unmasked_addons):
    """ A helper function to evaluate the performance of a particular recommendation
    strategy on a client with a set of installed addons that have been patially masked.
    Keyword arguments:
    recommendations_list -- a list of tuples containing (guid, confidence) pairs.

    unmasked_addons -- a list of the true installed addons for a test
    client. Each unmasked addon is expressed as a GUID.
    """
    # Organizer function to extract weights from recommendation list for passing to cllr.
    lrs_on_target_helper = [item[1] for item in recommendations_list if item[0] in unmasked_addons]
    lrs_off_target_helper = [item[1] for item in recommendations_list if item[0] not in unmasked_addons]
    return cllr(lrs_on_target_helper, lrs_off_target_helper)


""" End TAAR Ensemble math code starts """


"""TAAR Data loader code below """


def load_training_from_telemetry(spark):
    """ load some training data from telemetry given a sparkContext
    """
    sc = spark.sparkContext
    # Define the set of feature names to be used in the donor computations.
    AMO_DUMP_BUCKET = 'telemetry-parquet'
    AMO_DUMP_KEY = 'telemetry-ml/addon_recommender/addons_database.json'

    def load_amo_external_whitelist():
        """ Download and parse the AMO add-on whitelist.
        The json fetched here is generated by a weekly job that queries the AMO public API
        this whitelist is sure to exclude any Mozilla studies that are not correclty filtered
        as well as personal/unsigned/unlisted addons manually installed.
        :raises RuntimeError: the AMO whitelist file cannot be downloaded or contains
                              no valid add-ons.
        """
        final_whitelist = []
        amo_dump = {}
        try:
            # Load the most current AMO dump JSON resource.
            s3 = boto3.client('s3')
            s3_contents = s3.get_object(Bucket=AMO_DUMP_BUCKET, Key=AMO_DUMP_KEY)
            amo_dump = json.loads(s3_contents['Body'].read())
        except Exception:
            log.exception("Failed to download from S3", extra={
                "bucket": AMO_DUMP_BUCKET,
                "key": AMO_DUMP_KEY})

        # If the load fails, we will have an empty whitelist, this may be problematic.
        for key, value in amo_dump.items():
            addon_files = value.get('current_version', {}).get('files', {})
            # If any of the addon files are web_extensions compatible, it can be recommended.
            if any([f.get("is_webextension", False) for f in addon_files]):
                final_whitelist.append(value['guid'])

        if len(final_whitelist) == 0:
            raise RuntimeError("Empty AMO whitelist detected")

        return set(final_whitelist)

    def get_initial_sample():
        # noqa: ignore=E127,E502,E999
        """ Takes an initial sample from the longitudinal dataset
        (randomly sampled from main summary). Coarse filtering on:
        - number of installed addons
        - corrupt and generally wierd telemetry entries
        - isolating release channel
        - column selection
        """
        client_features_frame = spark.sql("SELECT * FROM longitudinal") \
                                     .where("active_addons IS NOT null")\
                                     .where("size(active_addons[0]) > 2")\
                                     .where("size(active_addons[0]) < 100")\
                                     .where("normalized_channel = 'release'")\
                                     .where("build IS NOT NULL AND build[0].application_name = 'Firefox'")
        client_features_frame = client_features_frame.selectExpr(
                    "client_id as client_id",
                    "active_addons[0] as active_addons",
                    "geo_city[0] as geo_city",
                    "subsession_length[0] as subsession_length",
                    "settings[0].locale as locale",
                    "os as os",
                    "places_bookmarks_count[0].sum AS bookmark_count",
                    "scalar_parent_browser_engagement_tab_open_event_count[0].value AS tab_open_count",
                    "scalar_parent_browser_engagement_total_uri_count[0].value AS total_uri",
                    "scalar_parent_browser_engagement_unique_domains_count[0].value AS unique_tlds")
        return client_features_frame

    def get_addons_per_client(users_df, minimum_addons_count):
        """ Extracts a DataFrame that contains one row
        for each client along with the list of active add-on GUIDs.
        """
        def is_valid_addon(guid, addon):
            """ Filter individual addons out to exclude, system addons,
            legacy addons, disabled addons, sideloaded addons.
            """
            return not (
                addon.is_system or
                addon.app_disabled or
                addon.type != "extension" or
                addon.user_disabled or
                addon.foreign_install or
                guid not in broadcast_amo_whitelist.value
            )
        # may need addiitonal whitelisting to remove shield addons

        # Create an add-ons dataset un-nesting the add-on map from each
        # user to a list of add-on GUIDs. Also filter undesired add-ons.
        return (
            users_df.rdd
            .map(lambda p: (p["client_id"],
                 [guid for guid, data in p["active_addons"].items() if is_valid_addon(guid, data)]))
            .filter(lambda p: len(p[1]) > minimum_addons_count)
            .toDF(["client_id", "addon_ids"])
        )

    log.info("Init loading client features")
    client_features_frame = get_initial_sample()
    log.info("Loaded client features")

    amo_white_list = load_amo_external_whitelist()
    log.info("AMO White list loaded")

    broadcast_amo_whitelist = sc.broadcast(amo_white_list)
    log.info("Broadcast AMO whitelist success")

    addons_info_frame = get_addons_per_client(client_features_frame, 4)
    log.info("Filtered clients with only 4 addons")

    taar_training = addons_info_frame.join(client_features_frame, 'client_id', 'inner').drop('active_addons')
    log.info("JOIN completed on TAAR training data")

    return taar_training


""" End TAAR Data loader code below """


class XValidator:
    """
    The xvalidator class will take in a dictionary of the form:

        client_id => {
            "geo_city": profile_data.get("city", ''),
            "subsession_length": profile_data.get("subsession_length", 0),
            "locale": profile_data.get('locale', ''),
            "os": profile_data.get("os", ''),
            "installed_addons": addon_ids,
            "disabled_addons_ids": profile_data.get("disabled_addons_ids", []),
            "bookmark_count": profile_data.get("places_bookmarks_count", 0),
            "tab_open_count": profile_data.get("scalar_parent_browser_engagement_tab_open_event_count", 0),
            "total_uri": profile_data.get("scalar_parent_browser_engagement_total_uri_count", 0),
            "unique_tlds": profile_data.get("scalar_parent_browser_engagement_unique_domains_count", 0),
        }

    We want to splice the inbound data into N number of folds to do a
    standard cross validation where one fold is used to test the
    training of the model based on the other N-1 folds of data.

    The additional constraint we have is that the test dataset must
    mask some of the addons in the 'installed_addons' list.  This is
    so that the recommender has a chance to 'fill' in the masked
    addons.
    """

    def __init__(self, n_folds, addons_minsize):
        assert n_folds > 1
        self._n_folds = n_folds
        self._addons_minsize = addons_minsize

    def cross_validation_split(self, dataset):
        """Split a dataset into k folds
        """
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / self._n_folds)
        for fold_i in range(self._n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(0, len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)

        return dataset_split

    def mask_addons(self, dataslice):
        """Iterate over the dataslice and return a list of masked
        client json blobs, as well as a map of addon_mask_id->list
        of masked addons.

        Note that each of the client_data blobs in the dataset
        does *not* contain a client ID.  This is ok for us, as we
        only care to reconcile the masked installed addons with the
        predicted addons.  We don't care about reconciling all the
        way back to the original client ID
        """

        masked_addons_by_clientid = {}
        new_dataslice = []
        for idx, client_data in enumerate(dataslice):
            client_data['addon_mask_id'] = idx

            # use a random selection of addons
            installed_addon_set = list(client_data['installed_addons'])

            keep_set = set()
            for i in range(self._addons_minsize):
                idx = randrange(0, len(installed_addon_set))
                keep_set.add(installed_addon_set.pop(idx))
            masked_addon_set = set(installed_addon_set) - keep_set

            masked_addons_by_clientid[idx] = list(masked_addon_set)
            client_data['installed_addons'] = list(keep_set)
            new_dataslice.append(client_data)

        return (new_dataslice, masked_addons_by_clientid)

    def check_predicted_addons(self,
                               client_data,
                               predicted_addons,
                               masked_addons_by_clientid):
        """
        Check the predicted addons for a singular client.

        Return a 3-tuple of :

        (prediction_accuracy_rate, expected_addons_set, correctly_predicted_addon_set)
        """
        addon_mask_id = client_data['addon_mask_id']

        expected_addons = set(masked_addons_by_clientid[addon_mask_id])
        predicted_set = set(predicted_addons)
        match_set = expected_addons.intersect(predicted_set)

        prediction_rate = len(match_set) / len(expected_addons) * 1.0
        return prediction_rate, expected_addons, match_set


def main(spark):
    dataset = load_training_from_telemetry(spark)
    dataset = dataset.sample(0.000001)

    # This will be a sample from telemetry.

    n_folds = 3
    scores = evaluate_algorithm(dataset, n_folds, MIN_ADDONSETSIZE=2)

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
