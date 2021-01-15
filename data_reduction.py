import sys
import boto3

#path = sys.argv[-1]
#path = "s3://sagemaker-aidevops/inference-output/output.txt"
#path = "s3://sagemaker-aidevops/inference-output/output.txt"
bucket_name = 'aidevops-inference-pipeline-bucket'
key_read ='inference-data/sample.jpg'
key_write ='working-storage/sample.jpg'
#pred_output = 'inference-output/output.txt'
pred_output = 'output-artifacts/output.txt'


def upload_glacier(obj_to_upload):
    glacier_client = boto3.client('glacier')    
    id = glacier_client.upload_archive(vaultName='adas', body=obj_to_upload['Body'].read())
    print('Uploaded to location: ',  id['location'])

def delete_from_landing(bucket_name, key):
    print('Deleting from landing...')
    s3resource = boto3.resource("s3")
    obj = s3resource.Object(bucket_name, key)
    response = obj.delete()
    print(response['DeleteMarker'])

def copy_to_working(bucket_name, key_read, key_write):
    s3r = boto3.resource('s3')
    bucket = s3r.Bucket(bucket_name)
    copy_source = {
          'Bucket': bucket_name,
          'Key': key_read
        }
    bucket.copy(copy_source, key_write)


#result_file = open(path,"r")
#result = result_file.readline()
#print('Result is:', result)
#result_file.close()

s3_client = boto3.client('s3')

obj = s3_client.get_object(Bucket= bucket_name,Key=pred_output)

result_byte = obj['Body'].read()
result = result_byte.decode("utf-8")
float_result = float(result)


print("Output artifact read value ="+ result)

read_obj = s3_client.get_object(Bucket=bucket_name, Key=key_read)

if (float_result < 0.5):
    # Push to Glacier
    print ('Pushing to Glacier')
    upload_glacier(read_obj)
else:
    # Push to working storage
    print ('Pushing to Working storage')
    copy_to_working(bucket_name, key_read, key_write)

#delete_from_landing(bucket_name, key_read)
