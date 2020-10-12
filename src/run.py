"""
Runs an experiment from a config.
"""
from argparse import ArgumentParser
import numpy as np
import os
from shutil import copyfile
import torch
from tqdm import tqdm
import yaml

from data import *
from models import *
from regimen import train_seq2seq, train_lm
import reporter
from truncation_models import *
from truncation_regimen import train_truncation
import utils


# Mapping from data.dataset_type to dataset class
dataset_lookup = {
    "dyckmk": Dataset,
    "dyckkm_ending": DatasetDyckEnding,
    "scan": DatasetSCAN,
    "mt": DatasetMT,
}

# Mapping from lm.lm_type to model class
seq2seq_lookup = {
    "seq2seq": Seq2SeqLSTM,
    "s2s_lookup": Seq2SeqPrecomputed,
    "s2s_lookup_hs": Seq2SeqPrecomputedHiddenStates,
}

# Mapping from truncation.model_type to model class
truncator_lookup = {
    "oracle": TruncationOracle,
    "dyckkm_ending": DyckKMProbePredictor,
}

def run_lm(args, dataset):
    """
    Runs a language model based on args parsed from the config.

    There are two steps to running. First, if the user passes the `--train-lm`
    flag, the model will be trained. If the user passes the `--train-truncation`
    flag we will train the probe to predict where EOS tokens end.  Next, the
    model will be evaluated. The data used for training and evaluation is
    encapsulated in the `dataset` argument.

    Args:
        args: Arguments parsed from the yaml config file.
        dataset: a data.Dataset object containing the training, validation,
            and test examples
    """
    lm = RNNLM(args)

    # Step 1: train LM
    if args['train'] and not isinstance(dataset, DatasetDyckEnding):
        lm.train()
        train_lm(args, lm, dataset)

    model_path = None
    if os.path.isfile(args['lm']['lm_path']):
        model_path = args['lm']['lm_path']
    lm.load_pretrained_model(model_path=model_path)
    lm.eval()

    if isinstance(dataset, DatasetDyckEnding):
        # create truncation model dir if it does not exist
        model_dir = utils.get_trunc_model_path_of_args(args)
        if not model_dir.endswith(".params"):
            os.makedirs(model_dir, exist_ok=True)

        truncator = truncator_lookup[args['truncation']['model_type']](args, lm)
        if args['train-truncation']:
            truncator.train()
            train_truncation(args, truncator, dataset)
        truncator.load_pretrained_model()
        truncator.eval()

        reporter.evaluate_lm(args, lm, dataset, 'dev', truncator)
        reporter.evaluate_lm(args, lm, dataset, 'test', truncator)
    else:
        # Step 2: Run evaluation
        reporter.evaluate_lm(args, lm, dataset, 'dev')
        reporter.evaluate_lm(args, lm, dataset, 'test')

def run_seq2seq(args, dataset):
    """
    Runs a squence-to-sequence based on args parsed from the config.

    There are three steps to running. First, if the user passes the `--train-lm`
    flag, the model will be trained. Second, if the user passes the
    `train-truncation` flag, a truncation model will be trained. Third, the 
    trained models will be data used for training and evaluation is
    encapsulated in the `dataset` argument.

    Args:
        args: Arguments parsed from the yaml config file.
        dataset: a data.Dataset object containing the training, validation,
            and test examples
    """
    # Search for what type of seq2seq model to use (e.g. RNN, Transformer) 
    seq2seq_model_class = seq2seq_lookup[args['lm']['lm_type']]
    lm = seq2seq_model_class(args)
    if args['lm']['lm_type'] in ('s2s_lookup', 's2s_lookup_hs'):
        lm.load_from_data(dataset)

    # Step 1: train seq2seq model
    if args['train-seq2seq']:
       lm.train()
       train_seq2seq(args, lm, dataset)

    # for consistency, after training, load the newly trained model from disk
    # the model can also be specified in the config directly by path
    model_path = None
    if os.path.isfile(args['lm']['lm_path']):
        model_path = args['lm']['lm_path']
    lm.load_pretrained_model(model_path=model_path)
    lm.eval()

    # Step 2: train truncation model
    truncator_cls = truncator_lookup[args['truncation']['model_type']]
    if truncator_cls.requires_lm:
        truncator = truncator_lookup[args['truncation']['model_type']](args, lm)
    else:
        truncator = truncator_lookup[args['truncation']['model_type']](args)

    if args['train-truncation']:
        truncator.train()
        train_truncation(args, truncator, dataset)
    truncator.load_pretrained_model()
    truncator.eval()

    # Step 3: Run evaluation
    reporter.evaluate_seq2seq(args, lm, dataset, truncator, 'dev')
    reporter.evaluate_seq2seq(args, lm, dataset, truncator, 'test')

def main():
    argp = ArgumentParser()
    argp.add_argument('config')
    argp.add_argument('--train', action="store_true")
    argp.add_argument('--train-seq2seq', action="store_true")
    argp.add_argument('--train-truncation', action="store_true")
    argp_args = argp.parse_args()
    args = yaml.safe_load(open(argp_args.config))
    args['train-seq2seq'] = argp_args.train_seq2seq
    args['train-truncation'] = argp_args.train_truncation
    args['train'] = argp_args.train or argp_args.train_truncation or argp_args.train_seq2seq
    args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set random seed
    torch.manual_seed(args['lm']['seed'])
    np.random.seed(args['lm']['seed'])

    # prepare results and model directories
    output_dir = utils.get_results_dir_of_args(args)
    tqdm.write("Writing results to {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    copyfile(argp_args.config, "{}/{}".format(output_dir, "config.yaml"))

    # seq2seq
    model_dir = utils.get_lm_path_of_args(args)
    if not model_dir.endswith(".params"):
        os.makedirs(model_dir, exist_ok=True)

    # Search for dataset
    dataset = dataset_lookup[args['data']['dataset_type']](args)

    # Run whatever experiment necessary
    if args['lm']['lm_type'] == 'rnnlm':
        run_lm(args, dataset)
    elif args['lm']['lm_type'] in seq2seq_lookup:
        # truncation
        model_dir = utils.get_trunc_model_path_of_args(args)
        if not model_dir.endswith(".params"):
            os.makedirs(model_dir, exist_ok=True)
        run_seq2seq(args, dataset)


if __name__ == '__main__':
    main()
