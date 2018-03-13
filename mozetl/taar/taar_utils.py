import boto3
import json
import logging

from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AMO_DUMP_BUCKET = 'telemetry-parquet'
AMO_DUMP_KEY = 'telemetry-ml/addon_recommender/addons_database.json'


def write_to_s3(source_file_name, s3_dest_file_name, s3_prefix, bucket):
    """Store the new json file containing current top addons per locale to S3.

    :param source_file_name: The name of the local source file.
    :param s3_dest_file_name: The name of the destination file on S3.
    :param s3_prefix: The S3 prefix in the bucket.
    :param bucket: The S3 bucket.
    """
    client = boto3.client('s3', 'us-west-2')
    transfer = boto3.s3.transfer.S3Transfer(client)

    # Update the state in the analysis bucket.
    key_path = s3_prefix + s3_dest_file_name
    transfer.upload_file(source_file_name, bucket, key_path)


def store_json_to_s3(json_data, base_filename, date, prefix, bucket):
    """Saves the JSON data to a local file and then uploads it to S3.

    Two copies of the file will get uploaded: one with as "<base_filename>.json"
    and the other as "<base_filename><YYYYMMDD>.json" for backup purposes.

    :param json_data: A string with the JSON content to write.
    :param base_filename: A string with the base name of the file to use for saving
        locally and uploading to S3.
    :param date: A date string in the "YYYYMMDD" format.
    :param prefix: The S3 prefix.
    :param bucket: The S3 bucket name.
    """
    FULL_FILENAME = "{}.json".format(base_filename)

    with open(FULL_FILENAME, "w+") as json_file:
        json_file.write(json_data)

    archived_file_copy =\
        "{}{}.json".format(base_filename, date)

    # Store a copy of the current JSON with datestamp.
    write_to_s3(FULL_FILENAME, archived_file_copy, prefix, bucket)
    write_to_s3(FULL_FILENAME, FULL_FILENAME, prefix, bucket)


class AddonChecks:

    MIN_STAR_RATING = 3

    _method_cache = None

    def __init__(self):
        checks = [k for k in self.__class__.__dict__.keys() if k.startswith("check_")]
        self._method_cache = [getattr(self, method_name)
                              for method_name in checks]

    def check_is_webextension(self, extension_meta):
        addon_files = extension_meta.get('current_version', {}).get('files', {})
        # If any of the addon files are web_extensions compatible, it can be recommended.
        return any([f.get("is_webextension", False) for f in addon_files])

    def check_min_rating(self, extension_meta):
        rating = extension_meta.get('rating', {}).get('average', 0)
        return rating >= self.MIN_STAR_RATING

    def verify_meta(self, extension_meta):
        for method in self._method_cache:
            if not method(extension_meta):
                return False
        return True


def load_amo_external_whitelist():
    """ Download and parse the AMO add-on whitelist.

    :raises RuntimeError: the AMO whitelist file cannot be downloaded or contains
                          no valid add-ons.
    """

    # If the load fails, we will have an empty whitelist, this may
    # be problematic.
    amo_dump = {}
    try:
        # Load the most current AMO dump JSON resource.
        s3 = boto3.client('s3')
        s3_contents = s3.get_object(Bucket=AMO_DUMP_BUCKET, Key=AMO_DUMP_KEY)
        byte_data = s3_contents['Body'].read()
        with open('/tmp/amo_dump.json', 'wb') as fout:
            fout.write(byte_data)
        amo_dump = json.loads(byte_data.decode('utf8'))
    except ClientError:
        logger.exception("Failed to download from S3", extra={
            "bucket": AMO_DUMP_BUCKET,
            "key": AMO_DUMP_KEY})

    final_whitelist = []
    checker = AddonChecks()
    for _ignored, extension_meta in amo_dump.items():
        if checker.verify_meta(extension_meta):
            final_whitelist.append(extension_meta['guid'])

    if len(final_whitelist) == 0:
        raise RuntimeError("Empty AMO whitelist detected")

    return final_whitelist
