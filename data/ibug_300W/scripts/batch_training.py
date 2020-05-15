import argparse
import json
import logging
import multiprocessing
import sys
from datetime import datetime

import dlib


def train(
    path_input_xml: str,
    path_output_model: str,
    options: dlib.shape_predictor_training_options,
):
    dlib.train_shape_predictor(path_input_xml, path_output_model, options)


def config_to_training_options(config: dict):
    result = dlib.shape_predictor_training_options()

    for option, value in config.items():
        if not value:
            continue
        if hasattr(result, option):
            setattr(result, option, value)

    return result


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input-xml",
        dest="input",
        required=True,
        help="path to input training XML file",
        type=str,
    )
    ap.add_argument(
        "-c",
        "--config",
        dest="config",
        required=True,
        help="Batch training config file",
        type=str,
    )
    ap.add_argument(
        "-o",
        "--output-dir",
        dest="output",
        required=True,
        help="Output directory for trained dlib shape predictor models.",
        type=str,
    )
    ap.add_argument(
        "-t",
        "--threads",
        dest="threads",
        required=False,
        help="number of threads to use for training. Defaults to CPU count.",
        type=int,
    )
    parsed_args = vars(ap.parse_args())

    return parsed_args


TIMESTAMP_START = get_timestamp()


class TrainingData:
    def __init__(self, config: dict):
        self.config = config

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def options(self) -> dlib.shape_predictor_training_options:
        return config_to_training_options(self.config)

    @property
    def json(self) -> str:
        return json.dumps(self.config)


def setup_logger():
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)

    handler_file = logging.FileHandler(
        "{}/{}_batch-training.log".format(args["output"], TIMESTAMP_START)
    )
    handler_file.setLevel(logging.DEBUG)
    logger_root.addHandler(handler_file)

    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(logging.DEBUG)
    logger_root.addHandler(handler_stdout)


if __name__ == "__main__":
    # Parse command line options
    args = parse_args()
    input_labels_xml = args["input"]
    output_model_dir = args["output"]
    if args["threads"]:
        target_thread_count = args["threads"]
    else:
        target_thread_count = multiprocessing.cpu_count()

    setup_logger()

    # Prepare training data
    with open(args["config"], "r") as config_stream:
        training_config_list = json.load(config_stream)

    training_data_list = []
    for training_config in training_config_list:
        training_data_list.append(TrainingData(config=training_config))

    # Train models sequentially
    for training_data in training_data_list:
        model_name = "{}_{}".format(TIMESTAMP_START, training_data.name)
        output_model_path = model_name + ".dat"

        training_options = training_data.options
        training_options.num_threads = target_thread_count
        logging.info("Training options:\n{}".format(training_options))

        # Save model-specific config to output dir
        current_config_path = "{}/{}.json".format(output_model_dir, model_name)
        with open(current_config_path, "w") as output_stream:
            output_stream.write(training_data.json)

        # Start training
        train(input_labels_xml, output_model_path, training_options)
