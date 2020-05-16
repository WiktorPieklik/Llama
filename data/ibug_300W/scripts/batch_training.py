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
        "--train-xml",
        dest="train_xml",
        required=True,
        help="path to training labels XML file",
        type=str,
    )
    ap.add_argument(
        "--test-xml",
        dest="test_xml",
        required=True,
        help="path to test labels XML file",
        type=str,
    )
    ap.add_argument(
        "--config-json",
        dest="config_json",
        required=True,
        help="Batch training config file",
        type=str,
    )
    ap.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Output directory for trained dlib shape predictor models.",
        type=str,
    )
    ap.add_argument(
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
        if not self.name:
            raise ValueError("Config name must be specified and unique.")

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
    log_format =" %(asctime)s :: %(levelname)-8s :: %(message)s"
    date_format = "%Y-%m-%d_%H-%M-%S"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt=date_format,
    )
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)

    handler_file = logging.FileHandler(
        "{}/{}_batch-training.log".format(args["output_dir"], TIMESTAMP_START)
    )

    handler_file.setLevel(logging.DEBUG)
    formatter_file_handler = logging.Formatter(
        fmt=log_format, datefmt=date_format
    )
    handler_file.setFormatter(formatter_file_handler)
    logger_root.addHandler(handler_file)


if __name__ == "__main__":
    # Parse command line options
    args = parse_args()
    fpath_train_labels_xml = args["train_xml"]
    fpath_output_model_dir = args["output_dir"]
    fpath_test_labels_xml = args["test_xml"]
    fpath_config_json = args["config_json"]
    if args["threads"]:
        target_thread_count = args["threads"]
    else:
        target_thread_count = multiprocessing.cpu_count()

    setup_logger()

    # Prepare training data
    with open(fpath_config_json, "r") as config_stream:
        training_config_list = json.load(config_stream)

    training_data_list = []
    for training_config in training_config_list:
        current_name = training_config["name"]
        if not current_name:
            raise ValueError("Config name cannot be None.")
        if training_config["name"] in [conf["name"] for conf in training_config_list]:
            raise ValueError(
                "Name '{}' already occurred in another training config. "
                "Config names must be unique.".format(current_name)
            )
        training_data_list.append(TrainingData(config=training_config))

    # Train models sequentially
    for trained_count, training_data in enumerate(training_data_list):
        logging.info("========== Training {} ==========".format(trained_count))
        model_name = "{}_{}".format(TIMESTAMP_START, training_data.name)
        output_model_path = model_name + ".dat"

        training_options = training_data.options
        training_options.num_threads = target_thread_count
        training_options.be_verbose = True
        logging.info(str(training_options))

        # Save model-specific config to output dir
        current_config_path = "{}/{}.json".format(fpath_output_model_dir, model_name)
        with open(current_config_path, "w") as output_stream:
            output_stream.write(training_data.json)

        # Start training
        train(fpath_train_labels_xml, output_model_path, training_options)

        # Evaluate resulting model
        print("Evaluating trained model...")
        # error_train = dlib.test_shape_predictor(
        #     fpath_train_labels_xml, output_model_path
        # )
        error_test = dlib.test_shape_predictor(fpath_test_labels_xml, output_model_path)
        # logging.info("{} on TRAIN labels => {} MAE.".format(model_name, error_train))
        logging.info("{} on TEST labels => {} MAE.".format(model_name, error_test))
