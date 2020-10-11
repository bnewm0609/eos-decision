"""
Runs the training and generation script for OpenNMT
"""
from argparse import ArgumentParser
import os
import subprocess
import torch
import yaml

PREPROCESSING_CMD = "python OpenNMT-py/onmt/bin/preprocess.py -train_src {} -train_tgt {} -valid_src {} -valid_tgt {} -save_data {}"
TRAIN_CMD = "python OpenNMT-py/onmt/bin/train.py {}"
TRANSLATE_CMD = "python OpenNMT-py/onmt/bin/translate.py {}"
EVALUATE_CMD = "python src/run.py {}"

def load_args():
    ap = ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--train-seq2seq", action='store_true')
    ap_args = ap.parse_args()

    args = yaml.safe_load(open(ap_args.config))
    args['train'] = ap_args.train_seq2seq
    args['config'] = ap_args.config
    return args

def should_run_preprocessing(args):
    """
    Checks if preprocessing script should be run by seeing if the .vocab.pt
    file exists.
    """
    use_eos = args['lm']['use_eos']
    args = args['onmt']['preprocessing']
    data_dir = args['data_dir']
    if use_eos:
        data_dir = os.path.join(data_dir, 'onmt_eos')
    else:
        data_dir = os.path.join(data_dir, "onmt_no_eos")
    vocab_file = os.path.join(data_dir, ".vocab.pt")
    return not os.path.exists(vocab_file)

def run_preprocessing(args):
    """
    Runs OpenNMT preprocessing script
    """
    use_eos = args['lm']['use_eos']
    args = args['onmt']['preprocessing']
    train_src = os.path.join(args['data_dir'], args['train_src'])
    train_tgt = os.path.join(args['data_dir'], args['train_tgt'])
    dev_src = os.path.join(args['data_dir'], args['dev_src'])
    dev_tgt = os.path.join(args['data_dir'], args['dev_tgt'])
    data_dir = args['data_dir']
    extra_args = ""
    if use_eos:
        data_dir = os.path.join(data_dir, 'onmt_eos')
    else:
        extra_args = " -disable_eos_sampling"
        data_dir = os.path.join(data_dir, "onmt_no_eos")
    cmd = PREPROCESSING_CMD.format(train_src, train_tgt, dev_src, dev_tgt, data_dir)
    cmd += extra_args
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def should_train(args):
    """
    Checks if the model should be trained.
    """
    return args['train']

def form_arg_str(args, taboo_fields=set()):
    """
    Creates command line string from args dict.
    """
    args_str = []
    for arg_name, arg_val in args.items():
        if arg_name in taboo_fields:
            continue
        if isinstance(arg_val, bool):
            args_str.append(f"-{arg_name}")
        else:
            args_str.append(f"-{arg_name} {arg_val}")
    return args_str

def train(args):
    """
    Runs OpenNMT train script.
    """
    data_dir = args['onmt']['preprocessing']['data_dir']
    if args['lm']['use_eos']:
        data_dir = os.path.join(data_dir, "onmt_eos")
    else:
        data_dir = os.path.join(data_dir, "onmt_no_eos")
    train_args = args['onmt']['training']
    model_dir = train_args['save_model']
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    args_str = [f'-data {data_dir}']
    args_str += form_arg_str(train_args)
    args_str = " ".join(args_str)
    cmd = TRAIN_CMD.format(args_str)
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

TRANSLATE_TABOO_FIELDS = {'dev_src', 'test_src', 'model'}
def translate(args):
    """
    Runs OpenNMT translate script
    """
    disable_eos_sampling = args['lm'].get('disable_eos_sampling', False)
    data_dir = args['onmt']['preprocessing']['data_dir']
    model_path = args['onmt']['training']['save_model']
    model_path = os.path.join(model_path, args['onmt']['translate']['model'])
    output_paths = (args['lm']['dev_output_path'], args['lm']['test_output_path'])
    data_paths = (os.path.join(data_dir, args['onmt']['translate']['dev_src']),
                  os.path.join(data_dir, args['onmt']['translate']['test_src']))

    for data_path, output_path in zip(data_paths, output_paths):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        args_str = form_arg_str(args['onmt']['translate'], TRANSLATE_TABOO_FIELDS)
        args_str += [f'-src {data_path}', f'-output {output_path}', f'-model {model_path}']
        if disable_eos_sampling:
            args_str.append("-disable_eos_sampling")
        args_str = " ".join(args_str)
        cmd = TRANSLATE_CMD.format(args_str)
        print(f"> {cmd}")
        subprocess.run(cmd, shell=True, check=True)

def evaluate(args):
    """
    Runs custom evaluation script
    """
    cmd = EVALUATE_CMD.format(args['config'])
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    # load in the yaml config with all of the args
    args = load_args()

    # if we have already run the preprocessing scripts, .vocab and .train and .val should already exist
    # so don't run them again.
    if should_run_preprocessing(args):
        print("Running preprocessing script")
        run_preprocessing(args)
    else:
        print("Found preprocessed data!")

    # Relatedly, only train the model if the the command line arg "--train-seq2seq" was passed.
    if should_train(args):
        train(args)
    else:
        print("Skipping training.")

    # Then, get predicted sequences from the model on the dev data and test data
    translate(args)

    # Finally, evaluate the model on dev/test (this involves calling our old friend `run.py`)
    # evaluate(args) # DON'T run at same time because this one doesn't need gpu
                     # (and we don't want to take up the GPU unecessarily)

if __name__ == "__main__":
    main()
