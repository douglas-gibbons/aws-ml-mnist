#!/usr/bin/env python3

"""
  Downloads NMIST data from  http://yann.lecun.com/exdb/mnist/
  and saves it as a python pickle file in the format of the MNIST data class
  found in data.py
"""

import pickle, gzip
import urllib.request
from pathlib import Path
import numpy as np
from data import MNIST
import sklearn.utils

from config import * # Read in config variables


# See http://yann.lecun.com/exdb/mnist/ for data description
urls =  [
  "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
  "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
]



for url in urls:
  filename = url.split('/')[-1]
  if not Path(filename).is_file():
    print('Downloading',filename)
    urllib.request.urlretrieve(url, filename)

def read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
  print('Extracting', filename)
  with gzip.open(filename,'rb') as f:
    magic_number = read32(f)
    num_images = read32(f)
    rows = read32(f)
    cols = read32(f)
    print('Number of images:',num_images,'Rows:',rows,'Cols:',cols)
    buf = f.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def extract_labels(filename):
  print('Extracting', filename)
  with gzip.open(filename,'rb') as f:
    magic_number = read32(f)
    num_labels = read32(f)
    print('Number of labels:',num_labels)
    buf = f.read(num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8)
    # Convert to "one-hot" vectors
    
    return (labels)

    
train_images = extract_images('train-images-idx3-ubyte.gz')
test_images = extract_images('t10k-images-idx3-ubyte.gz')
train_labels = extract_labels('train-labels-idx1-ubyte.gz')
test_labels = extract_labels('t10k-labels-idx1-ubyte.gz')

# Randmoze the order of training and test data
(train_images, train_labels) = sklearn.utils.shuffle(train_images, train_labels, random_state=0)
(test_images, test_labels) = sklearn.utils.shuffle(test_images, test_labels, random_state=0)


print('train images:',train_images.shape)
print('train labels:',train_labels.shape)
print('test images:',test_images.shape)
print('test labels:',test_labels.shape)

data = MNIST(train_images,test_images,train_labels,test_labels)
  
pickle.dump(data, open( pickle_file, "wb" ) ,protocol = 4 )
	
