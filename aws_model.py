#!/usr/bin/env python3

"""
  Create ML model and evaluates it using the MNIST data uploaded from 
  aws_upload.py
"""

import numpy as np
from data import MNIST
# from matplotlib import pyplot as plt
import base64
import boto3
import json
import os
import sys
import random
import string

from config import * # Read in config variables


"""
  Creates all the objects needed to build an ML Model & evaluate its quality.
"""
def build_model(schema_fn, recipe_fn, name):

  ml = boto3.client('machinelearning')
  train_ds_id = 'ds-train-'+project_id
  test_ds_id = 'ds-test-'+project_id
  train_data_s3_url = 's3://' + data_bucket + '/' + train_data_s3_file
  test_data_s3_url = 's3://' + data_bucket + '/' + test_data_s3_file
  
  # Train data Source
  create_data_source(
    ml, train_ds_id, train_data_s3_url, schema_fn, "MNIST Train data"
  )
  
  # Test data source
  create_data_source(
    ml, test_ds_id, test_data_s3_url, schema_fn, "MNIST Test data"
  )
                
  ml_model_id = create_model(ml, train_ds_id, recipe_fn, name)
  eval_id = create_evaluation(ml, ml_model_id, test_ds_id, name)

  return ml_model_id


def create_data_source(ml, ds_id, data_s3_url, schema_fn, name):
  
  spec = {
      "DataLocationS3": data_s3_url,
      "DataSchema": open(schema_fn).read(),
  }
  ml.create_data_source_from_s3(
      DataSourceId=ds_id,
      DataSpec=spec,
      DataSourceName=name,
      ComputeStatistics=True
  )
  print("Created data set %s" % ds_id)


def create_model(ml, train_ds_id, recipe_fn, name):

  model_id = 'ml-' + project_id
  ml.create_ml_model(
      MLModelId=model_id,
      MLModelName=name + " model",
      MLModelType="MULTICLASS",
      Parameters={
          # Refer to the "Machine Learning Concepts" documentation
          # for guidelines on tuning your model
          "sgd.maxPasses": "100",
          "sgd.maxMLModelSizeInBytes": "104857600",  # 100 MiB
          "sgd.l2RegularizationAmount": "1e-4",
      },
      Recipe=open(recipe_fn).read(),
      TrainingDataSourceId=train_ds_id
  )
  print("Created ML Model %s" % model_id)
  return model_id


def create_evaluation(ml, model_id, test_ds_id, name):
  eval_id = 'ev-' + project_id
  ml.create_evaluation(
      EvaluationId=eval_id,
      EvaluationName=name + " evaluation",
      MLModelId=model_id,
      EvaluationDataSourceId=test_ds_id
  )
  print("Created Evaluation %s" % eval_id)
  return eval_id


schema_fn = "image.schema"
recipe_fn = "recipe.json"
name = "MNIST"

model_id = build_model(schema_fn, recipe_fn, name)

