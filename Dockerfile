FROM ubuntu:18.04  
#python:3.6

# set the working directory
RUN ["mkdir", "app"]
WORKDIR "app"

#install build dependencies
RUN apt-get update && apt-get install -y git 
RUN apt-get install -y libmysqlclient-dev mysql-client
RUN apt-get install -y python-pip
RUN apt-get install -y curl
RUN apt-get install -y unzip
# install code dependencies
COPY "requirements.txt" .
RUN ["pip", "install", "-r", "requirements.txt"]

RUN PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip && curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$PROTOC_ZIP && unzip -o $PROTOC_ZIP -d /usr/local bin/protoc &&  unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && rm -f $PROTOC_ZIP

RUN git clone https://github.com/tensorflow/models.git
RUN protoc ./models/research/object_detection/protos/string_int_label_map.proto --python_out=.
RUN cp -R models/research/object_detection/ object_detection/

#COPY "BatchCreator.ipynb" .
COPY codebuild-prediction.py /app/codebuild-prediction.py
COPY codebuild-objectdetection.py /app/codebuild-objectdetection.py


# install environment dependencies
ENV SHELL /bin/bash


# VOLUME ["/home/tarun/Downloads" "/tmp"]
# If the following dependency is put in requirements, it gets ignored as 1.0.8 is already found but that version poses issues in code
#RUN ["pip", "install", "--upgrade", "Keras-Applications==1.0.7"]
#COPY BatchCreator.py /app/BatchCreator.py
#COPY "Train.py" /app/Train.py

#COPY "run.sh" .
#RUN ["chmod", "+x", "./run.sh"]
#COPY training-data/ /tmp/training-data/
#RUN ["mkdir", "-p" , "/app/training-data"]
#COPY training-data/* /app/training-data/

#COPY "preparemodel.py" /app/preparemodel.py
#ENTRYPOINT ["./run.sh"]
#CMD ["train"]
