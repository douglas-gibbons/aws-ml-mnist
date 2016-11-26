"""
  Global configuration
"""

project_id = 'MNIST-1' # Change this id to keep multiple runs in AWS 
data_bucket = "YOUR_BUCKET_NAME"
bucket_region = 'us-east-2'
train_data_s3_file = "data/train_data.csv"
test_data_s3_file = "data/test_data.csv"
output_data = "output/"
pickle_file = "data.pkl"
prediction_output_filename = 'bp-' + project_id + '-test_data.csv.gz'
threshold = 0.77
