#!/usr/bin/env python3

"""
  Uploads MNIST data to S3.  
  Run download.py before this to create the local pickle file
"""

import numpy as np
from data import MNIST
import pickle
import boto3

from config import * # Read in config variables


# Load the data
data = pickle.load( open(pickle_file, 'rb') )

print('train images:',data.train_images.shape)
print('train labels:',data.train_labels.shape)
print('test images:',data.test_images.shape)
print('test labels:',data.test_labels.shape)

# reshape labals to 2D array
train_labels = data.train_labels.reshape( data.train_labels.shape[0],1)
test_labels = data.test_labels.reshape( data.test_labels.shape[0],1)

# Add the data and labels together
train_data = np.concatenate(( data.flatten(data.train_images),train_labels) ,axis=1)
test_data = np.concatenate(( data.flatten(data.test_images),test_labels) ,axis=1)


# Create CSV files
np.savetxt('train_data.csv', train_data, fmt='%d', delimiter=',') 
np.savetxt('test_data.csv', test_data, fmt='%d', delimiter=',') 


# Save data to S3
s3 = boto3.client('s3',bucket_region)

print('Uploding files')
s3.upload_file('train_data.csv', data_bucket, train_data_s3_file)
s3.upload_file('test_data.csv', data_bucket, test_data_s3_file)
print('Complete')
