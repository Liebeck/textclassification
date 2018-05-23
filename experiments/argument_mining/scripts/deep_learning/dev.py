import argparse
import json
import logging
import sys
from experiments.argument_mining.scripts.deep_learning import benchmark


def config_logger(log_level=logging.INFO):
    logger = logging.getLogger('')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(log_level)


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining Deep Learning')
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-padding_length', type=int, help='Padding length of each input sequence', default=20)
    argparser.add_argument('-batch_size', type=int, default=32)
    argparser.add_argument('-keras_model_name', type=str, default="blstm")
    argparser.add_argument('-epochs', type=int, default=5)
    argparser.add_argument('-embeddings_cache_name', type=str, default=None)
    argparser.add_argument('-keras_model_parameters', type=str, default=None)
    argparser.add_argument('-evaluation_ID', type=str, default=666)
    return argparser.parse_args()


if __name__ == '__main__':
    config_logger(log_level=logging.INFO)
    logger = logging.getLogger()
    arguments = config_argparser()
    if arguments.keras_model_parameters:  # Parse keras model parameters from a JSON string into a dictionary
        arguments.keras_model_parameters = json.loads(arguments.keras_model_parameters)
    benchmark.benchmark(subtask=arguments.subtask, config_parameters=vars(arguments))
