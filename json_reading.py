__author__ = 'Mat'

import json

json_data = open('data.json').read()
data = json.loads(json_data)


print data["dataset"]["data"][3][2]