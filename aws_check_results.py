#!/usr/bin/env python3

"""
  Checks an AWS ML batch prediction output file
  (for MNIST).
  
"""

import boto3
import gzip
import sys
import numpy as np
from config import * # Read in config variables


filename = prediction_output_filename

# Download the file from S3
s3_file = output_data + 'batch-prediction/result/' + filename
print('Retrieving',s3_file, 'from', data_bucket)

s3_client = boto3.client('s3',bucket_region)
s3_client.download_file(data_bucket, s3_file, filename)


# Tally up the results

headings = None
hits = misses = 0

with gzip.open(filename,'rb') as f:
  
  for line in  f.readlines():
    line = line.decode('utf-8').rstrip()
    vals = line.split(',')
    # First line is the header row
    if not headings:
      headings = vals[1:]
    else:
      # Find the column with the max value and pick the corrisonding
      # value from the header row
      trueLabel = vals[0]
      guesses = np.array(vals[1:]).astype(np.float32)
      guessLabel = headings[guesses.argmax()]
      if guessLabel == trueLabel: hits += 1
      else: misses += 1

      
print('Hits: %d Misses: %d Score: %.2f%%' % (hits, misses, hits * 100/(hits+misses) )  )

