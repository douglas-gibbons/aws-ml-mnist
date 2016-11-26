#!/usr/bin/env python3

import numpy as np
from data import MNIST
# from matplotlib import pyplot as plt
import boto3
import json
import os
import sys
import random
import string
import datetime
import time
  

from config import * # Read in config variables

test_data_s3_url = 's3://' + data_bucket + '/' + test_data_s3_file
s3_output_url = 's3://' + data_bucket + '/' + output_data


def use_model(model_id, threshold, schema_fn, output_s3, data_s3url):
    """Creates all the objects needed to build an ML Model & evaluate its quality.
    """
    ml = boto3.client('machinelearning') 

    poll_until_completed(ml, model_id)  # Can't use it until it's COMPLETED
    #ml.update_ml_model(MLModelId=model_id, ScoreThreshold=threshold)
    #print("Set score threshold for %s to %.2f" % (model_id, threshold))

    bp_id = 'bp-' + project_id
    ds_id = 'ds-test-' + project_id
    ml.create_batch_prediction(
        BatchPredictionId=bp_id,
        BatchPredictionName="Batch Prediction MNIST",
        MLModelId=model_id,
        BatchPredictionDataSourceId=ds_id,
        OutputUri=s3_output_url
    )
    print("Created Batch Prediction %s" % bp_id)


def poll_until_completed(ml, model_id):
    delay = 2
    while True:
        model = ml.get_ml_model(MLModelId=model_id)
        status = model['Status']
        message = model.get('Message', '')
        now = str(datetime.datetime.now().time())
        print("Model %s is %s (%s) at %s" % (model_id, status, message, now))
        if status in ['COMPLETED', 'FAILED', 'INVALID']:
            break

        # exponential backoff with jitter
        delay *= random.uniform(1.1, 2.0)
        time.sleep(delay)


use_model('ml-' + project_id, threshold, "image.schema", s3_output_url, test_data_s3_url)



