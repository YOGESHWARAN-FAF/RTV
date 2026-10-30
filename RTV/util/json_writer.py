import json


def json_wrtiter(data_dict, path):
    with open(path, "w") as outfile:
        json.dump(data_dict, outfile)
