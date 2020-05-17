import json


def load_json(json_path: str) -> dict:
    with open(json_path, "r") as json_file:
        return json.load(json_file)
