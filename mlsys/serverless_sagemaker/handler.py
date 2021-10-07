import uuid
from typing import Dict, Any
import time
import json
import os


# read in the params for regression
BETA = float(os.environ['BETA'])
INTERCEPT = float(os.environ['INTERCEPT'])


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


def run_regression(Xs: list) -> list:
    """
    For each input, we run a regression as 

    y = BETA * X + INTERCEPT
    """
    if not Xs:
        return None

    return [INTERCEPT + (BETA * x)  for x in Xs]


def sagemaker_regression(event, context):
    """
    This function is reacheable as a GET endpoint thanks to AWS magic!
    
    You can call this with:

    https://XXX.execute-api.us-west-2.amazonaws.com/dev/simple_regression?x=10
    """
    # start a timer
    start = time.time()
    # print this for debug
    print("Received event: {}".format(json.dumps(event)))
    # read parameters
    params = event.get('queryStringParameters', {})
    # get Xs as a list from a parameter called x
    Xs = [float(x) for x in params['x'].split(',')] if 'x' in params else None
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