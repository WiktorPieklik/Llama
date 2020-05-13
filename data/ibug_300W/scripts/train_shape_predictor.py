# USAGE
# python train_shape_predictor.py --training ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml --model eye_predictor.dat

# import the necessary packages
import multiprocessing
import argparse
import dlib


def train(
    path_input_xml: str,
    path_output_model: str,
    options: dlib.shape_predictor_training_options,
):
    dlib.train_shape_predictor(path_input_xml, path_output_model, options)

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="path to input training XML file",
        type=str,
    )
    ap.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        help="path serialized dlib shape predictor model",
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
    args = vars(ap.parse_args())

    # grab the default options for dlib's shape predictor
    print("[INFO] setting shape predictor options...")
    training_options = dlib.shape_predictor_training_options()

    # define the depth of each regression tree -- there will be a total
    # of 2^tree_depth leaves in each tree; small values of tree_depth
    # will be *faster* but *less accurate* while larger values will
    # generate trees that are *deeper*, *more accurate*, but will run
    # *far slower* when making predictions
    training_options.tree_depth = 4

    # regularization parameter in the range [0, 1] that is used to help
    # our model generalize -- values closer to 1 will make our model fit
    # the training data better, but could cause overfitting; values closer
    # to 0 will help our model generalize but will require us to have
    # training data in the order of 1000s of data points
    training_options.nu = 0.1

    # the number of cascades used to train the shape predictor -- this
    # parameter has a *dramtic* impact on both the *accuracy* and *output
    # size* of your model; the more cascades you have, the more accurate
    # your model can potentially be, but also the *larger* the output size
    training_options.cascade_depth = 15

    # number of pixels used to generate features for the random trees at
    # each cascade -- larger pixel values will make your shape predictor
    # more accurate, but slower; use large values if speed is not a
    # problem, otherwise smaller values for resource constrained/embedded
    # devices
    training_options.feature_pool_size = 400

    # selects best features at each cascade when training -- the larger
    # this value is, the *longer* it will take to train but (potentially)
    # the more *accurate* your model will be
    training_options.num_test_splits = 50

    # controls amount of "jitter" (i.e., data augmentation) when training
    # the shape predictor -- applies the supplied number of random
    # deformations, thereby performing regularization and increasing the
    # ability of our model to generalize
    training_options.oversampling_amount = 5

    # amount of translation jitter to apply -- the dlib docs recommend
    # values in the range [0, 0.5]
    training_options.oversampling_translation_jitter = 0.1

    # tell the dlib shape predictor to be verbose and print out status
    # messages our model trains
    training_options.be_verbose = True

    # number of threads/CPU cores to be used when training -- we default
    # this value to the number of available cores on the system, but you
    # can supply an integer value here if you would like
    if args["threads"]:
        target_thread_count = args["threads"]
    else:
        target_thread_count = multiprocessing.cpu_count()

    training_options.num_threads = target_thread_count

    # log our training options to the terminal
    print("[INFO] shape predictor options:")
    print(training_options)

    # train the shape predictor
    print("[INFO] training shape predictor...")
    train(args["input"], args["output"], training_options)
