"""Test suite for taar_lite_guidguid Job."""

import json
import boto3
import pytest
import mock
from moto import mock_s3
from mozetl.taar import taar_lite_guidguid, taar_utils
from pyspark.sql import Row
from mozetl.taar.taar_utils import store_json_to_s3, load_amo_external_whitelist
from dummy_spark import RDD
from dummy_spark import SparkContext, SparkConf

sconf = SparkConf()
sc = SparkContext(master='', conf=sconf)


"""
Expected schema of co-installation counts dict.
| -- key_addon: string(nullable=true) 
| -- coinstallation_counts: array(nullable=true) 
| | -- element: struct(containsNull=true) 
| | | -- id: string(nullable=true) 
| | | -- n: long(nullable=true)
"""
class MockRDF:
    def __init__(self, rdd):
        self.rdd = rdd


class TaarRDD:
    def __init__(self, rdd):
        self._rdd = rdd

    def toDF(self, *args):
        result = [Row(**dict(zip(args[0], row))) for row in self.__dict__['_rdd']._jrdd]
        result = RDD(result, sc)
        result = MockRDF(result)
        return result

    def __getattr__(self, attr):
        func = getattr(self._rdd, attr)
        def curry_func(*args, **kwargs):
            def wrap_with_taar():
                result = func(*args, **kwargs)
                if isinstance(result, RDD):
                    result = TaarRDD(result)
                return result
            return wrap_with_taar()
        return curry_func

MOCK_TELEMETRY_SAMPLE = MockRDF(TaarRDD(RDD([
    Row(installed_addons=["test-guid-1", "test-guid-2", "test-guid-3"]),
    Row(installed_addons=["test-guid-1", "test-guid-3"]),
    Row(installed_addons=["test-guid-1", "test-guid-4"]),
    Row(installed_addons=["test-guid-2", "test-guid-5", "test-guid-6"]),
    Row(installed_addons=["test-guid-1", "test-guid-1"])
], sc)))

MOCK_ADDON_INSTALLATIONS = {
    "test-guid-1":
        {"test-guid-2": 1,
         "test-guid-3": 2,
         "test-guid-4": 2
         },
    "test-guid-2":
        {"test-guid-1": 2,
         "test-guid-5": 1,
         "test-guid-6": 1
         }}

MOCK_KEYED_ADDONS = [
    Row(key_addon='test-guid-1',
        coinstalled_addons=['test-guid-2','test-guid-3', 'test-guid-4']),
    Row(key_addon='test-guid-1',
        coinstalled_addons=['test-guid-3','test-guid-4']),
    Row(key_addon="test-guid-2",
        coinstalled_addons=['test-guid-1','test-guid-5', 'test-guid-6']),
    Row(key_addon="test-guid-2",
        coinstalled_addons=['test-guid-1'])
    ]



@mock.patch('mozetl.taar.taar_lite_guidguid.load_training_from_telemetry',
            return_value=MOCK_TELEMETRY_SAMPLE)
@mock_s3
def test_load_training_from_telemetry(spark):
    # Sanity check that mocking is happening correctly.
    assert taar_lite_guidguid.load_training_from_telemetry(spark) == MOCK_TELEMETRY_SAMPLE

    rdf = taar_lite_guidguid.load_training_from_telemetry(spark)

    rdd1 = rdf.rdd
    import pdb
    pdb.set_trace()
    rdd2 = rdd1.flatMap(lambda x: taar_lite_guidguid.key_all(x.installed_addons))
    dframe = rdd2.toDF(['key_addon', "coinstalled_addons"])
    assert dframe == MOCK_KEYED_ADDONS


def test_addon_keying():
    assert taar_lite_guidguid.key_all(MOCK_KEYED_ADDONS) == MOCK_ADDON_INSTALLATIONS
