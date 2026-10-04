with open('src/aws/cloudwatch_collector.py', 'r') as f:
    c = f.read()

c = c.replace(
    'Dimensions=[{"Name": "TargetGroup", "Value": TG_ARN.split(":")[-1]}],',
    'Dimensions=[{"Name": "LoadBalancer", "Value": ALB_ARN.split("loadbalancer/")[1]}],'
)

with open('src/aws/cloudwatch_collector.py', 'w') as f:
    f.write(c)
