# aws-ml-mnist

AWS Machine Learning Example with MNIST data


This is an example of using AWS ML to make predictions on the famous ["MNIST" dataset](ttp://yann.lecun.com/exdb/mnist/).

The code is an example of how to run AWS Machine Learning with Python.


### Pre-requisites

The code uses python v3, and will need the following libraries:

* boto3
* numpy

You'll also need full access to an AWS account and some access keys

It was tested on a Ubuntu 16.04 machine, but it should run on most systems with Python 3.

### Setting Up

Clone the code.

Create and AWS S3 bucket and set up a bucket policy. The policy should allow access to AWS ML components and to the scripts to upload and download files. An example policy (which will need editing) can be found in the code directory, and is called "bucket_policy.json".

Create directories "data" and "output" inside the bucket

Set up ~/.aws/credentials with credentials and a region suitable for ML. For example:

```
[default]
aws_access_key_id = YOUR_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
region=us-east-1
```

Edit ```config.py``` and change the data_bucket name and check the bucket region is correct.

### Running the Code

The various python files are designed to run in order:

1. Download the MNIST data locally by running ```./download_mnist.py```
1. Format the data and upload to S3 ```./aws_upload.py```
1. Build and run the AWS ML model ```./aws_model.py```
1. Wait for the model to complete building by checking the [web interface](https://console.aws.amazon.com/machinelearning/home?region=us-east-1#/).
1. Run ```./aws_use_model.py``` to use the model to make predictions on the test images
1. Wait for the predictions to complete (again, check the [web interface](https://console.aws.amazon.com/machinelearning/home?region=us-east-1#/)).
1. Download the prediction output and evaluate the predictions ```./aws_check_results.py```

The output should show the accuracy of the machine learning predictions to be about 91%.
