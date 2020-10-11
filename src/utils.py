"""
Utilities for determining paths to corpora, results, models
given config dictionaries describing an experiment, as well
as other miscillaneous functions.
"""

from argparse import ArgumentParser
import copy
import os
import re
import string
import torch
from tqdm import tqdm
import yaml

# Constants
DETOKENIZER_FILE = "tools/detokenizer.perl"

def get_identifier_iterator():
  ids = iter(list(string.ascii_lowercase))
  k = 1
  while True:
    try:
      str_id = next(ids)
    except StopIteration:
      ids = iter(list(string.ascii_lowercase))
      k += 1
      str_id = next(ids)
    yield str_id*k


def get_vocab_of_bracket_types(bracket_types):
    id_iterator = get_identifier_iterator()
    ids = [next(id_iterator) for x in range(bracket_types)]
    vocab = {x: c for c, x in enumerate(['(' + id_str for id_str in ids] + [id_str + ')' for id_str in ids] + ['START', 'END'])}
    return vocab, ids


def load_config_from_path(path, train=False):
    """
    Loads a yaml config from the path and adds additional arguments.

    Args:
        path: the path of the config to load
        train (bool):  whether the command line args specify to begin training
    """
    args = yaml.safe_load(open(path))
    args['train'] = train
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args

def load_config_from_cmdline():
    """
    Loads a yaml config file from the command line.

    Some scripts need to do this, and it's cleaner to have them call
    this method than to copy and past from run.py.

    Returns:
        dict containing arguments specified in yaml file
    """
    argp = ArgumentParser()
    argp.add_argument('config')
    argp.add_argument('--train', action="store_true")
    argp_args = argp.parse_args()
    return load_config_from_path(argp_args.config, argp_args.train)


def get_corpus_paths_of_args(args):
    """
    Takes a (likely yaml-defined) argument dictionary
    and returns the paths of the train/dev/test
    corpora files.
    """
    args = copy.deepcopy(args)
    if 'vocab_size' in args['language']:
        del args['language']['vocab_size']
    if (args['corpus']['train_override_path'] or
        args['corpus']['dev_override_path'] or
        args['corpus']['test_override_path']):
        train_path = args['corpus']['train_override_path']
        dev_path = args['corpus']['dev_override_path']
        test_path = args['corpus']['test_override_path']
    else:
        path = os.path.join(
         args['corpus']['root'],
        '-'.join([
          '_'.join([key+str(value) for key, value in args['language'].items()])]))
        path = re.sub('max_stack_depth', 'msd', path)
        path = re.sub('learning_rate', 'lr', path)
        path = re.sub('sample_count', 'sc', path)
        path = re.sub('max_length', 'ml', path)
        path = re.sub('train', 'tr', path)
        path = re.sub('test', 'te', path)
        path = re.sub('num_layers', 'nl', path)
        path = re.sub('hidden_dim', 'hd', path)
        path = re.sub('embedding_dim', 'ed', path)
        path = re.sub('analytic_model', 'am', path)
        path = re.sub('max_epochs', 'me', path)
        path = re.sub('batch_size', 'bs', path)
        path = re.sub('min_length', 'ml', path)
        path = re.sub('min_state_filter_percent', 'fp', path)
        path = re.sub('length_effects', 'le', path)
        train_path = os.path.join(path, 'train.formal.txt')
        dev_path = os.path.join(path, 'dev.formal.txt')
        test_path = os.path.join(path, 'test.formal.txt')
    return {'train':train_path, 'dev':dev_path, 'test':test_path}


def get_results_dir_of_args(args):
    """
    Takes a (likely yaml-defined) argument dictionary
    and returns the directory to which results of the
    experiment defined by the arguments will be saved
    """
    # Don't include these fields in the path
    taboo_fields = set(['load_finetuned', 'load_cache', 'lm_path', 'model_path',
        'dev_output_path', 'test_output_path'])
    path = os.path.join(
        args['reporter']['results_path'],
        '-'.join([
            args['data']['dataset_type'],
            args['data']['dataset_name'],
            '_'.join([key+str(value) for key, value in args['lm'].items() if key not in taboo_fields]),
            '_'.join([key+str(value) for key, value in args.get('truncation', {}).items() if key not in taboo_fields])
        ])
      )

    path = re.sub('dataset_type', 'ds', path)
    path = re.sub('dataset_name', '-', path)
    path = re.sub('lm_type', '', path)
    path = re.sub('batch_size', 'bs', path)
    path = re.sub('cell_type', 'cell', path)
    path = re.sub('False', 'F', path)
    path = re.sub('True', 'T', path)
    path = re.sub('decoder', 'dec', path)
    path = re.sub('encoder', 'enc', path)
    path = re.sub('calculate_neighbors', 'cn', path)
    path = re.sub('teacher_forcing_ratio', 'tfr', path)
    path = re.sub('bidirectional', 'bd', path)
    path = re.sub('attention', 'at', path)
    path = re.sub('use_eos', 'eos', path)
    path = re.sub('disable_eos_sampling', 'des', path)
    path = re.sub('dev_output_path', 'dop', path)
    path = re.sub('test_output_path', 'top', path)
    path = re.sub('max_seq_len', 'msl', path)

    return path


def get_lm_path_of_args(args):
    """
    Takes a (likely yaml-defined) argument dictionary
    and returns the directory to which results of the
    experiment defined by the arguments will be saved
    """

    # Don't include these fields in the path
    taboo_fields = set(['load_fine_tuned', 'cell_type',
                        'encoder_cell_type', 'decoder_cell_type',
                        'calculate_neighbors', 'lm_path',
                        'disable_eos_sampling', 'max_seq_len', 
                        'dev_output_path', 'test_output_path'])

    # hard code dataset name for dyck k :(
    dataset_type = args['data']['dataset_type']
    if dataset_type == "dyckkm_ending":
        dataset_type = "dyckmk"

    path = os.path.join( args['lm']['lm_path'],
        '-'.join([
            dataset_type,
            args['data']['dataset_name'],
            '_'.join([key+str(value) for key, value in args['lm'].items() if key not in taboo_fields])
        ])
      )

    path = re.sub('dataset_type', 'ds', path)
    path = re.sub('dataset_name', '-', path)
    path = re.sub('lm_type', '', path)
    path = re.sub('batch_size', 'bs', path)
    path = re.sub('False', 'F', path)
    path = re.sub('True', 'T', path)
    path = re.sub('decoder', 'dec', path)
    path = re.sub('encoder', 'enc', path)
    path = re.sub('teacher_forcing_ratio', 'tfr', path)
    path = re.sub('attention', 'at', path)
    path = re.sub('bidirectional', 'bd', path)
    path = re.sub('use_eos', 'eos', path)
    path = re.sub('dev_output_path', 'dop', path)
    path = re.sub('test_output_path', 'top', path)

    return path

def get_trunc_model_path_of_args(args):
    taboo_fields = ['model_path']
    
    path = os.path.join( args['truncation']['model_path'],
        '-'.join([
            args['data']['dataset_type'],
            args['data']['dataset_name'],
            '_'.join([key+str(value) for key, value in args['truncation'].items() if key not in taboo_fields]),
            'eos{}'.format(args['lm']['use_eos'])
        ])
      )

    path = re.sub('embed_dim', 'es', path)
    path = re.sub('hidden_dim', 'hs', path)
    path = re.sub('encoder_num_layers', 'nl', path)
    path = re.sub('bidirectional', 'bd', path)
    path = re.sub('attention', 'at', path)
    path = re.sub('dropout', 'do', path)
    path = re.sub('True', 'T', path)
    path = re.sub('False', 'F', path)

    return path

def reserve_alot_of_memory():
    """From https://discuss.pytorch.org/t/reserving-gpu-memory/25297/2"""
    total, used = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split(",")

    total = int(total)
    used = int(used)

    max_mem = int(total * 0.8)
    tqdm.write("Reserving {} MiB for pytorch".format(max_mem))
    block_mem = max_mem - used

    x = torch.rand((256,1024,block_mem)).cuda()
    x = torch.rand((2,2)).cuda()


def generate_masks(input_lens, max_len=-1, device=None):
    """
    Generates integer mask for sequences of given lengths.

    The mask is used for computing attention. A value of 0 means attention
    should be computed at that position, and a value of 1 means attention
    should not be computed for that position (e.g. because the token at
    that position is a padding token).

    Args:
        input_lens: torch.Tensor(batch_size) containing the lengths of the
            inputs sequences of a batch.
    Returns:
        torch.Tensor(batch_size, max_input_len). A mask tensor whose values
        are described above.
    """
    if device is None:
        device = input_lens.device
    if max_len < 0:
        max_len = input_lens.max().item()
    masks = torch.zeros((len(input_lens), max_len)).to(device)
    for i in range(len(input_lens)):
        masks[i, input_lens[i]:] = 1
    return masks


# Misc. functions
def identity(*args):
    """A do nothing function"""
    return tuple(args)

