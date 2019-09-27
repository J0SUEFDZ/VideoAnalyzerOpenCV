import json
from analyzer import Detector


def lambda_handler(event, context):
    data_retrieved = event['Records'][0]['s3']
    bucket_name = data_retrieved['bucket']['name']
    file_name = data_retrieved['object']['key']
    detector = Detector(file_name, bucket_name)
    result = detector.detect_movement()
    return {
        'statusCode': 200,
        'filename': file_name
    }
