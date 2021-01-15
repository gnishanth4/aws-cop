import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel
import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np
import os, io
import cv2
import boto3
import math
#import imread
#from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import RMSprop,SGD
import tensorflow as tf
from keras.models import model_from_json
#from imageai.Detection import ObjectDetection
import pandas as pd
import csv
import uuid

max_res = 86.0414
max_out = 7.9576
min_res = 13.669
min_out = 1.0987
name=['name']

# downloading image from S3.
#bucketname = 'sagemaker-aidevops' # replace with your bucket name
bucketname =  'aidevops-inference-pipeline-bucket' # replace with your bucket name

filename = 'inference-data/sample.jpg' # replace with your object key
s3 = boto3.resource('s3')
s3.Bucket(bucketname).download_file(filename, 'sample.jpg')


client = boto3.client('sagemaker')
end_point=client.list_endpoints(StatusEquals='InService',SortBy='CreationTime')['Endpoints'][0]['EndpointName']

#print(client.list_training_jobs(SortBy='CreationTime')['TrainingJobSummaries'][0]['TrainingJobName'])

predictor=sagemaker.tensorflow.model.TensorFlowPredictor(end_point, sagemaker.Session())

def image_colorfulness(image):
    #print("color",image)
    #image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

def pre_process_img(autoencoder, graph, image_file):
    predictedFile = "predictedImage.jpg"
    residualFile = "residual.jpg"
    originalFile = "originalImage.jpg"
    objectDetectedFile = "objectDetected.jpg"
    #image_binary_data = image_file.read()
    #np_img = np.fromstring(image_binary_data, np.uint8)
    #image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image = image_file
    
    C = image_colorfulness(image)
    g = cv2.resize(image, (320, 60))
    g = g.reshape(-1, 60, 320, 3)
    g = g / 255.
    with graph.as_default():
        pred = autoencoder.predict(g)
        
    #values = pred['outputs']['score']['float_val']
    values = pred['predictions']
    arr = np.asarray(values).reshape(1,60,320,3)
    
    residual = g.reshape(1,60, 320, 3) - arr
    
    residual_val = np.sum((g.astype("float")  - arr.astype("float")) ** 2)
    
    residual_f= np.round(residual_val/C,4)
    if (residual_f < min_res):
        residual_f = min_res
    if (residual_f > max_res):
        residual_f = max_res
    residual_f = (residual_f - min_res) / (max_res - min_res)
    uniqueID = uuid.uuid4().hex
    #detector.detectObjectsFromImage(input_image=originalFile, output_image_path=objectDetectedFile, minimum_percentage_probability=30,  extract_detected_objects=True)
    
    return residual_f, uniqueID

# The sample model expects an input of shape [1,50]
#data = np.random.randn(1, 50)
graph = tf.get_default_graph()


image_file = cv2.imread("sample.jpg")
residual_f, uniqueID = pre_process_img(predictor, graph, image_file)
print("Prediction output: "+ str(residual_f))

s3 = boto3.resource('s3')

filename = 'output.txt'
with open(filename, "w+") as f:
    f.write(str(residual_f))

print("Saving prediction output file into s3")
#s3.upload_file(f,bucketname,"output-artifacts")
s3.Bucket(bucketname).upload_file(filename, "output-artifacts/output.txt")
print("file saved to s3")
