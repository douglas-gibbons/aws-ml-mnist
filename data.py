import numpy as np

class MNIST:
  train_images = False
  test_images = False
  train_labels = False
  test_labels = False
  
  def __init__(self,train_images,test_images,train_labels,test_labels):
    self.train_images = train_images
    self.test_images = test_images
    self.train_labels = train_labels
    self.test_labels = test_labels
    
  def one_hot_labels(self,labels):
    one_hot = [ [ n == label for n in range(10) ] for label in labels]
    one_hot = np.array(one_hot).astype(np.uint8)
    return one_hot
    
  """ Flattens a given set of images """
  def flatten(self,images):
    (num_images,width,height,col_layers) = images.shape
    return images.reshape(num_images,width*height)

