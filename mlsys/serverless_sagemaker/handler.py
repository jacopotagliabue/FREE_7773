import uuid
from typing import Dict, Any
import time
import json
import os
import boto3


# instantiate AWS client for invoking sagemaker endpoint
runtime = boto3.client('sagemaker-runtime')
SAGEMAKER_ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME')


def wrap_response(status_code: int,
                  body: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Small function to wrap the model response in an actual API response (status, body, headers).
    :param status_code: http status code
    :param body: dictionary containing model predictions
    :return:
    """
    return {
        'isBase64Encoded': False,
        'statusCode': status_code,
        'headers': {
            # this makes the function callable across domains!
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(body),
    }

def get_response_from_sagemaker(model_input: list,
                                endpoint_name: str,
                                content_type: str = 'application/json') -> list:
    # get raw response from sagemaker
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType=content_type,
                                       Body=json.dumps(model_input))
    # return the response body, properly decoded
    return json.loads(response['Body'].read().decode())

def run_regression(Xs: list) -> list:
    """
    Invoke regression model hosted on SageMaker
    """
    if not Xs:
        return None

    response = get_response_from_sagemaker(model_input=Xs,
                                           endpoint_name=SAGEMAKER_ENDPOINT_NAME,
                                           content_type='application/json')
    return response


def sagemaker_regression(event, context):
    """
    This function is reacheable as a GET endpoint thanks to AWS magic!
    
    You can call this with:

    https://XXX.execute-api.us-west-2.amazonaws.com/dev/sagemaker_regression?x=10
    """
    # start a timer
    start = time.time()
    # print this for debug
    print("Received event: {}".format(json.dumps(event)))
    # read parameters
    params = event.get('queryStringParameters', {})
    # get Xs as a list from a parameter called x
    Xs = [[float(x)] for x in params['x'].split(',')] if 'x' in params else None
    predictions = run_regression(Xs)
    # be civilized: wrap the response around some useful data
    response_body = {
        'data': {
            'predictions': predictions
        },
        'metadata': {
            'eventId': str(uuid.uuid4()),
            'serverTimestamp': round(time.time() * 1000), # current epoch in millisec
            "time": time.time() - start
        }
    }

    # return response to the client
    return wrap_response(status_code=200, body=response_body)