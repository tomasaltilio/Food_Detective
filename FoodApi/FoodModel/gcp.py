import os

from google.cloud import storage
from termcolor import colored

BUCKET_NAME = "XXX"  # ⚠️ replace with your BUCKET NAME


def storage_upload(model_directory, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = '{}/{}/{}/{}'.format(
        'models',
        'food_detective_model',
        model_directory,
        'model.h5')
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.h5')
    print(colored("=> model.h5 uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.h5')