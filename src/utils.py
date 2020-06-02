import json


def json_file_to_dict(json_path: str) -> dict:
    with open(json_path, "r") as json_file:
        return json.load(json_file)
