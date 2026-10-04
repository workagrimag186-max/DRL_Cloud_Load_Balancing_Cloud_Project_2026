import boto3
from pprint import pprint
table = boto3.resource('dynamodb', region_name='us-east-1').Table('routing_decisions')
resp = table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key('pk').eq('STATE#live'), ScanIndexForward=False, Limit=1)
if resp['Items']:
    print("Latest State PK/SK:", resp['Items'][0]['pk'], resp['Items'][0]['sk'])
    print("CPU:", resp['Items'][0]['state_vector'][0:4])
    print("Instances found:", resp['Items'][0].get('metadata', {}).get('instances_found'))
else:
    print("No state found.")

resp_action = table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key('pk').eq('ACTION#live'), ScanIndexForward=False, Limit=1)
if resp_action['Items']:
    print("Latest Action PK/SK:", resp_action['Items'][0]['pk'], resp_action['Items'][0]['sk'])
    print("Action:", resp_action['Items'][0]['action'])
else:
    print("No action found.")
