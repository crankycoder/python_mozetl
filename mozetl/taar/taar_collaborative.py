# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import logging

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from .taar_utils import store_json_to_s3

import click
import datetime
import boto3
import json

AMO_BUCKET = "telemetry-parquet"
AMO_PREFIX = "telemetry-ml/addon_recommender/"
AMO_S3_FNAME = "extended_addons_database.json"

TOP200_S3_BUCKET = "telemetry-parquet"
TOP200_S3_PREFIX = "telemetry-ml/addon_recommender/"
TOP200_S3_FNAME = "only_guids_top_200.json"


logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

########
# Reimplementation of Java hashes


def _java_string_hashcode(s):
    # This is python re-implmentaion of java.lang.Object::hashCode()
    h = 0
    for c in s:
        h = (31 * h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def positive_hash(s):
    # the scala implementation of the collaborative recommender uses
    # a positive hash
    return _java_string_hashcode(s) & 0x7FFFFF


########


def read_from_s3(s3_dest_file_name, s3_prefix, bucket):
    """
    Read JSON from a S3 bucket and return the decoded JSON blob
    """

    full_s3_name = "{}{}".format(s3_prefix, s3_dest_file_name)
    conn = boto3.resource("s3", region_name="us-west-2")
    try:
        s3_obj = conn.Object(bucket, full_s3_name).get()
        s3_data = s3_obj["Body"].read()
        json_str = s3_data.decode("utf-8")
        stored_data = json.loads(json_str)
        logger.info("Loaded s3://{}/{}".format(bucket, full_s3_name))
    except Exception as e:
        msg = "Error decoding S3 bucket [{}][{}]: {}".format(
            bucket, full_s3_name, str(e)
        )
        logger.error(msg)
        raise e
    return stored_data


########


def get_df(spark, date_from, sampling=7):
    df = (
        spark.sql("select * from clients_daily")
        .where("client_id is NOT null")
        .where("active_addons IS NOT null")
        .where("channel = 'release'")
        .where("app_name = 'Firefox'")
        .where("submission_date_s3 >= '{}'".format(date_from))
        .where("sample_id < '{}'".format(sampling))
        .selectExpr(
            "client_id",
            "active_addons",
            "submission_date_s3",
            "row_number() OVER (PARTITION BY client_id ORDER BY submission_date_s3 desc) as rn",
        )
        .where("rn = 1")
        .drop("rn")
    )
    return df


def flatmapClosure(whitelist_bag, amo_db_bag):
    """ This function is applied to the clients data extract so that
    we get a DataFrame consisting of the 4-tuple :

    ( client_id,
      addonId,
      positive_hash(str(client_id)),
      positive_hash(str(addonId))
    )
    """

    def flatmapFunc(rdd_data):
        client_id = rdd_data["client_id"]
        for addon in rdd_data["active_addons"]:
            addonId = addon.addon_id.encode("utf8")
            if addonId in whitelist_bag and addonId in amo_db_bag:
                blocklisted = addon.blocklisted
                signedState = addon.signed_state
                userDisabled = addon.user_disabled
                appDisabled = addon.app_disabled
                addonType = getattr(addon, "type")
                isSystem = addon.is_system
                if (
                    not blocklisted
                    and (addonType != "extension" or signedState == 2)
                    and not userDisabled
                    and not appDisabled
                    and not isSystem
                ):
                    # Python3 and Python2 handle dictionary lookups
                    # differently.  Python3 is much more strict about
                    # looking up byte type keys in dictionaries where
                    # keys are strings.
                    if isinstance(client_id, bytes):
                        client_id = client_id.decode("utf8")
                        msg = "Fixed byte encoded client_id [{}]".format(client_id)
                        logger.info(msg)

                    if isinstance(addonId, bytes):
                        addonId = addonId.decode("utf8")
                        msg = "Fixed byte encoded addonId [{}]".format(addonId)
                        logger.info(msg)
                    yield [
                        client_id,
                        addonId,
                        positive_hash(str(client_id)),
                        positive_hash(str(addonId)),
                    ]

    return flatmapFunc


def get_ratings(spark, client_addons):
    # Build a dataframe of Rating tuples.  These are
    # (positive_hash(client_id), positive_hash(addon_id), 1.0F) to start
    def ratingsMapFunc(args):
        client_id, addon_id, hashed_client_id, hashed_addon_id = args
        return {"clientId": hashed_client_id, "addonId": hashed_addon_id, "rating": 1.0}

    return (
        client_addons.map(ratingsMapFunc)
        .repartition(spark.sparkContext.defaultParallelism)
        .toDF()
        .cache()
    )


def fit_ratings_model(ratings, max_iter):
    # Build the recommendation model using ALS on the training data
    # The old scala code used a custom NaNRegressionEvaluator which discarded NaN prediction values.
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics.
    #
    # See: https://spark.apache.org/docs/latest/ml-collaborative-filtering.html#cold-start-strategy
    #  Spark allows users to set the coldStartStrategy parameter to "drop" in order to drop any rows
    #  in the DataFrame of predictions that contain NaN values. The evaluation metric will then be
    #  computed over the non-NaN data and will be valid. Usage of this parameter is illustrated
    #  in the example below.

    als = ALS(
        seed=42,
        maxIter=max_iter,
        implicitPrefs=True,
        userCol="clientId",
        itemCol="addonId",
        ratingCol="rating",
        coldStartStrategy="drop",
    )

    evaluator = RegressionEvaluator(
        labelCol="rating", predictionCol="prediction", metricName="rmse"
    )

    # Note that parameters here are just what were used in the original
    # Scala implementation of the CollaborativeRecommender
    paramGrid = (
        ParamGridBuilder()
        .addGrid(als.rank, [10]) # [15, 25, 35])
        .addGrid(als.regParam, [0.01, 0.1])
        .addGrid(als.alpha, [1.0, 10, 20])
        .build()
    )

    cv = CrossValidator(
        estimator=als,
        evaluator=evaluator,
        estimatorParamMaps=paramGrid,
        numFolds=2,  # 10,
        parallelism=1,  # 20,
    )
    model = cv.fit(ratings)
    return model


def compute_serialized_mapping(addonMappingList, amo_db):
    """

    """
    tmpSerializedMapping = {}
    for addonId in addonMappingList:
        addon_meta = amo_db.get(addonId, None)
        if addon_meta is None:
            msg = "Can't find addonID [{}] in amo_db".format(addonId)
            logger.warn(msg)
            continue
        locale = addon_meta.get(
            "default_locale", ""
        )  # this is a mandatory field, empty should not happen
        preferred_name = addon_meta["name"].get(locale, "")
        _cur_version = addon_meta.get("current_version", {})
        _files = _cur_version.get("files", [])
        isWebextension = (
            sum(file_dict.get("is_webextension", False) for file_dict in _files) > 0
        )

        if preferred_name != "":
            if type(addonId) == bytes:
                addonId = addonId.decode("utf8")
            tmpSerializedMapping[positive_hash(addonId)] = {
                "name": preferred_name,
                "id": addonId,
                "isWebextension": isWebextension,
            }
    return tmpSerializedMapping


def transform_to_python_model(model):
    itemFactors = model.bestModel.itemFactors.collect()
    py_best_model = []
    for row in itemFactors:
        row = {"id": row.id, "features": row.features}
        py_best_model.append(row)
    return py_best_model


def load_json_to_s3(serializedMapping, best_model):
    # save the serialized mapping to S3
    date = datetime.date.today().strftime("%Y%m%d")
    store_json_to_s3(
        json.dumps(serializedMapping),
        "addon_mapping.new",
        date,
        "telemetry-ml/addon_recommender/",
        "telemetry-public-analysis-2",
    )

    # save the best model out to S3 as item_matrix
    store_json_to_s3(
        json.dumps(best_model),
        "item_matrix.new",
        date,
        "telemetry-ml/addon_recommender/",
        "telemetry-public-analysis-2",
    )


def transform(spark, df, whitelist_bag, amo_db_bag, amo_db, max_iter=20):
    # Transform and filter the client_addons dataframe with addon
    # metadata and addon whitelists
    # Construct the client_addons dataframe and collect all the data
    client_addons = df.rdd.flatMap(flatmapClosure(whitelist_bag, amo_db_bag))

    ratings = get_ratings(spark, client_addons)
    logger.info("Ratings generated")

    # Serialize add-on mapping and merge it with the AMODatabase to get a map of:
    # hashedClientID -> dict of addon metadata with keys (name, id, isWebExtension)
    #
    # Note that the lambda fucntion here extracts the un-hashed
    # addonID from the client_addons dataframe.
    addonMappingList = client_addons.map(lambda x: x[1]).distinct().cache().collect()

    # Coerce the elements of the list to bytes if necessary
    if type(addonMappingList[0]) == str:
        addonMappingList = [x.encode("utf8") for x in addonMappingList]

    logger.info("addonMappingList generated")

    serializedMapping = compute_serialized_mapping(addonMappingList, amo_db)
    logger.info("serializedMapping generated")

    model = fit_ratings_model(ratings, max_iter)
    logger.info("Model has been fitted")
    best_model = transform_to_python_model(model)
    logger.info("Best model computed.")
    return best_model, serializedMapping


def extract(spark, date, sample_rate):
    # Load S3 data
    # Load the latest whitelist of approved addons
    white_list = read_from_s3(TOP200_S3_FNAME, TOP200_S3_PREFIX, TOP200_S3_BUCKET)
    whitelist_bag = set([x.encode("utf8") for x in white_list])

    # Load the AMO database from the S3 JSON blob
    amo_db = read_from_s3(AMO_S3_FNAME, AMO_PREFIX, AMO_BUCKET)
    amo_db_bag = set([x.encode("utf8") for x in amo_db.keys()])

    df = get_df(spark, date)

    if sample_rate != 0:
        df = df.sample(False, sample_rate)

    return df, whitelist_bag, amo_db_bag, amo_db


@click.command()
@click.option("--date", required=True)
@click.option("--bucket", default="telemetry-private-analysis-2")
@click.option("--prefix", default="telemetry-ml/addon_recommender")
@click.option("--sample_rate", default=0.0, type=float)
def main(date, bucket, prefix, sample_rate):
    spark = (
        SparkSession.builder.appName("taar_collaborative")
        .enableHiveSupport()
        .getOrCreate()
    )

    df, whitelist_bag, amo_db_bag, amo_db = extract(spark, date, sample_rate)
    logger.info("Data extract completed")

    best_model, serializedMapping = transform(
        spark, df, whitelist_bag, amo_db_bag, amo_db
    )
    logger.info("Data transform completed")

    load_json_to_s3(serializedMapping, best_model)
    logger.info("Data load to S3 completed")
