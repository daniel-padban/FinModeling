import json

def read_json(path):
    with open(path,'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict