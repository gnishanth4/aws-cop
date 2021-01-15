#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
EDITING for iterative use

!git clone https://github.com/tensorflow/models.git

!protoc ./models/research/object_detection/protos/string_int_label_map.proto --python_out=.
!cp -R models/research/object_detection/ object_detection/
!rm -rf models


pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib 

pip install pandas

sudo yum install -y python-devel mysql-devel
pip install mysqlclient
pip install mysql
pip install pymysql
'''

import numpy as np
import os
import six.moves.urllib as urllib
#import six.moves as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import sys
import boto3

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def toDB(image_path, result):
    print('Writing to db :: ', image_path, "  :::  ", result )
    import pandas as pd
    import pymysql

    import sys
    #import ConfigParser
    import configparser
    # config object to read from file passed as an argument
    config = configparser.ConfigParser()
    config.read('config.ini')
    # input_file = open(sys.argv[2], "r")
    image_s3_path = image_path # input_file.readline().strip()
    image_objects = result # # input_file.readline().strip()

    host=config.get('connection','host')
    port=int(config.get('connection','port'))
    dbname= config.get('connection','dbname')
    user=config.get('connection','user')
    password=config.get('connection','password')

    read=config.get('flag','read_table')
    write=config.get('flag','write_table')
    # create a connection object
    conn = pymysql.connect(host, user=user,port=port,
                               passwd=password, db=dbname)


    # function to read records
    def read_table(config,conn):
        records= pd.read_sql(config.get('read_query','query'), con=conn) 
        print('Records thus far:')
        print(records)



    # function to insert records
    #sql = "INSERT INTO `working_metadata` (`image_s3_path`,`image_objects`) VALUES (%s, %s)"
    def write_table(config,conn):
        sql = config.get('write_query','query')
        cursor = conn.cursor()
        # Execute the query
        #cursor.execute(sql, (config.get('write_query','image_s3_path'),config.get('write_query','image_objects')))
        cursor.execute(sql, (image_s3_path,image_objects))
        # the connection is not autocommited by default. So we must commit to save our changes.
        conn.commit()
        #records= pd.read_sql(config.get('read_query','query'), con=conn)
        #print(records)
        #return records

    print(bool(read))
    if(read == 'True'):
        read_table(config,conn)

    if(write == 'True'):
        write_table(config,conn)


# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# model with more accurancy but up to you use a diferent model
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

#BUCKET = 'autonomous-mobility'
#KEY = 'testimages/sample.jpg'

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

from PIL import Image
from io import BytesIO

def getImage(key_string):
    s3_resource = boto3.client('s3')
    # key = 'working-storage/traffic.jpg'
    img_data = s3_resource.get_object(Bucket=BUCKET, Key=key_string)['Body'].read()
    image = Image.open(BytesIO(img_data))
    return BytesIO(img_data)


def getList(bucket_name_string, prefix_string):
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket_name_string)
    keys = []
    for my_bucket_object in my_bucket.objects.filter(Prefix=prefix_string):
        if my_bucket_object.key.endswith('jpg'):
            keys.append(my_bucket_object.key)
    #print(keys)
    return (keys)
# print (key.name for key in list(my_bucket.list('working-storage/', '/')))


def makeS3path(key):
    return 's3://' + BUCKET + '/' + key

def result(list_result, image_path):
    
    values = {}
    val_str = ''
    values = set(values)
    for i in objects:
        for key, value in i.items() :
            print(str(key))
            values.add(str(key))

    for s in values:
        val_str += s + ','

    val_str = val_str[:-1]
    return val_str


#BUCKET = 'sagemaker-aidevops'
BUCKET = 'aidevops-inference-pipeline-bucket'
prefix = 'working-storage/'
#TEST_IMAGE_PATHS = getList(BUCKET,prefix )
TEST_IMAGE_PATHS = ['working-storage/sample.jpg']
print('All paths in the bucket...')
for im in TEST_IMAGE_PATHS:
    print (makeS3path(im))


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
        print('working on ', makeS3path(image_path ))
        image = Image.open(getImage(image_path))
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        objects = []
        for index, value in enumerate(classes[0]):
            object_dict = {}
            if scores[0, index] > 0.6:
                object_dict[(category_index.get(value)).get('name').encode('utf8')] =                                 scores[0, index]
                objects.append(object_dict)
        print ('Processing result for ', image_path)
        toDB(makeS3path(image_path), result(objects, image_path))
        
print('Done!')


# In[ ]:




