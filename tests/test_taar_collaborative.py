"""Test suite for TAAR Ensemble Job."""

from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    LongType,
    BooleanType,
    ArrayType,
)

from mozetl.taar import taar_collaborative
import functools
import logging
import pytest
import random

random.seed(42)


logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

clientsdaily_schema = StructType(
    [
        StructField("submission_date_s3", StringType(), True),
        StructField("client_id", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("city", StringType(), True),
        StructField("subsession_hours_sum", LongType(), True),
        StructField("os", StringType(), True),
        StructField("app_name", StringType(), True),
        StructField("locale", StringType(), True),
        StructField(
            "active_addons",
            # active_addons is a list of dictionaries holding all
            # metadata related to an addon
            ArrayType(
                StructType(
                    [
                        StructField("addon_id", StringType(), True),
                        StructField("app_disabled", BooleanType(), True),
                        StructField("blocklisted", BooleanType(), True),
                        StructField("foreign_install", BooleanType(), True),
                        StructField("has_binary_components", BooleanType(), True),
                        StructField("install_day", LongType(), True),
                        StructField("is_system", BooleanType(), True),
                        StructField("is_web_extension", BooleanType(), True),
                        StructField("multiprocess_compatible", BooleanType(), True),
                        StructField("name", StringType(), True),
                        StructField("scope", LongType(), True),
                        StructField("signed_state", LongType(), True),
                        StructField("type", StringType(), True),
                        StructField("update_day", LongType(), True),
                        StructField("user_disabled", BooleanType(), True),
                        StructField("version", StringType(), True),
                    ]
                ),
                True,
            ),
        ),
        StructField("places_bookmarks_count_mean", LongType(), True),
        StructField(
            "scalar_parent_browser_engagement_tab_open_event_count_sum",
            LongType(),
            True,
        ),
        StructField(
            "scalar_parent_browser_engagement_total_uri_count_sum", LongType(), True
        ),
        StructField(
            "scalar_parent_browser_engagement_unique_domains_count_mean",
            LongType(),
            True,
        ),
    ]
)

default_sample = {
    "submission_date_s3": "20181220",
    "client_id": "client-id",
    "channel": "release",
    "city": "Boston",
    "subsession_hours_sum": 10,
    "os": "Windows",
    "app_name": "Firefox",
    "locale": "en-US",
    "active_addons": [],
    "places_bookmarks_count_mean": 1,
    "scalar_parent_browser_engagement_tab_open_event_count_sum": 2,
    "scalar_parent_browser_engagement_total_uri_count_sum": 3,
    "scalar_parent_browser_engagement_unique_domains_count_mean": 4,
}


# =========================================================================== #
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


@pytest.fixture()
def generate_data(dataframe_factory):
    return functools.partial(
        dataframe_factory.create_dataframe,
        base=default_sample,
        schema=clientsdaily_schema,
    )


@pytest.fixture
def extract_df(generate_data):

    sample_snippets = []
    for c_id in range(5):
        client_id = "client-id-{:02d}".format(c_id)
        variation = {
            "client_id": client_id,
            "city": "London",
            "subsession_hours_sum": 1107,
            "os": "MacOS",
            "locale": "en-UK",
            "active_addons": [],
        }

        # Pick 4 random addon GUIDs
        for addon_num in random.sample(range(1, 5), 4):
            addon_guid = "guid-{:05d}".format(addon_num)

            variation["active_addons"].append(
                {
                    "addon_id": addon_guid,
                    "blocklisted": False,
                    "user_disabled": False,
                    "app_disabled": False,
                    "signed_state": 2,
                    "type": "extension",
                    "foreign_install": False,
                    "is_system": False,
                }
            )

        sample_snippets.append(variation)

    dataframe = generate_data(sample_snippets)
    dataframe.createOrReplaceTempView("clients_daily")
    dataframe.cache()
    yield dataframe
    dataframe.unpersist()


@pytest.mark.timeout(90)
def test_transform_dataset(spark, extract_df):

    amo_db = {}
    for i in range(0, 500):
        guid = "guid-{:05d}".format(i).encode("utf8")
        meta = {
            "default_locale": "en-US",
            "name": {"en-US": "preferred name addon-{:05d}".format(i)},
            "current_version": {"files": [{"is_webextension": True}]},
        }
        amo_db[guid] = meta

    whitelist = amo_db_bag = amo_db
    collab = taar_collaborative.CollaborativeJob(spark)
    best_model, serializedMapping = collab.transform(
        extract_df, whitelist, amo_db_bag, amo_db, max_iter=5
    )

    for record in best_model:
        assert "id" in record
        assert "features" in record

    EXPECTED_MAPPING = {
        2282477: {
            "name": "preferred name addon-00001",
            "id": "guid-00001",
            "isWebextension": True,
        },
        2282480: {
            "name": "preferred name addon-00004",
            "id": "guid-00004",
            "isWebextension": True,
        },
        2282478: {
            "name": "preferred name addon-00002",
            "id": "guid-00002",
            "isWebextension": True,
        },
        2282479: {
            "name": "preferred name addon-00003",
            "id": "guid-00003",
            "isWebextension": True,
        },
    }

    assert EXPECTED_MAPPING == serializedMapping
