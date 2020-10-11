"""
Script for saving the hidden states from OpenNMT models if we want to run
PCA on them.
"""
from argparse import ArgumentParser
import glob
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import yaml

import onmt.inputters as inputters
from onmt.inputters.inputter import DatasetLazyIter 
from onmt.model_builder import build_model
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser as OnmtArgumentParser


def _get_parser():
    parser = OnmtArgumentParser(description='save_hs.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def get_dataset_paths(opt, corpus_type, eos):
    prefix = "onmt_eos" if eos else "onmt_no_eos"
    dataset_glob = opt.data + f'/{prefix}.' + corpus_type + '.[0-9]*.pt'
    dataset_paths = list(sorted(
        glob.glob(dataset_glob),
        key=lambda p: int(p.split(".")[-2])))
    return dataset_paths

def save_hidden_states(opt, args):
    OnmtArgumentParser.update_model_opts(opt)
    OnmtArgumentParser.validate_model_opts(opt)

    # load model
    model_path = os.path.join(opt.save_model, args['onmt']['translate']['model'])
    checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
    model_opt = OnmtArgumentParser.ckpt_model_opts(checkpoint["opt"])
    OnmtArgumentParser.update_model_opts(model_opt)
    OnmtArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    model = build_model(model_opt, opt, vocab, checkpoint)

    cache = []
    cache_idxs = [] # stores index into the training data and length of sentence
    for split in ('train', 'valid', 'test'):
        if split == 'test':
            test_file = args['data']['test_path']
            test_data = open(test_file, "r").readlines()
            test_src, test_tgt = zip(*[line.split("\t") for line in test_data])

            src_reader = inputters.str2reader["text"].from_opt(opt)
            tgt_reader = inputters.str2reader["text"].from_opt(opt)
            src_data = {"reader": src_reader, "data": test_src, "dir": None}
            tgt_data = {"reader": tgt_reader, "data": test_tgt, "dir": None}
            _readers, _data, _dir = inputters.Dataset.config(
                [('src', src_data), ('tgt', tgt_data)])

            data = inputters.Dataset(
                vocab, readers=_readers, data=_data, dirs=_dir,
                sort_key=inputters.str2sortkey["text"],
                filter_pred=None
            )

            batch_iter = inputters.OrderedIterator(
                dataset=data,
                device=args['device'],
                batch_size=64,
                batch_size_fn=None,
                train=False,
                sort=False,
                sort_within_batch=True,
                shuffle=False
            )
        else:
            train_dataset_paths = get_dataset_paths(opt, split, eos=args['lm']['use_eos'])

            batch_iter = DatasetLazyIter(
                    train_dataset_paths,
                    vocab, # vocab
                    64,    # batch size
                    None,  # "batch_fn"
                    1,     # "batch_size_multiple"
                    args['device'], # device
                    True,  # is train
                    8192,  # pool factor
                    repeat=False,
                    num_batches_multiple=1,
                    yield_raw_example=False)

        tgt_field = vocab["tgt"].base_field
        tgt_pad_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
        for batch_i, batch in tqdm(enumerate(batch_iter), desc=f"[{split}]"):
            # run through model
            if batch_i > 10000:
                break
            src, src_lengths = batch.src
            tgt = batch.tgt
            with torch.no_grad():
                hidden_states, attn = model(src, tgt, src_lengths, bptt=False, with_align=False)

            # save src idxs and hidden states
            pad_masks = (tgt[1:] != tgt_pad_idx).squeeze(2)
            cache.extend(hidden_states[pad_masks].cpu().numpy())
            cache_idxs.extend([(
                batch.indices[i].item(),
                pad_masks[:, i].sum().item(),
                ) for i in range(pad_masks.size(1))])

        cache = np.vstack(cache)
        # save the cache and the cache indices
        save_path = args['reporter']['results_path']
        print(save_path)
        print(cache.shape)
        np.save(os.path.join(save_path, f"cache.{split}.npy"), cache)
        with open(os.path.join(save_path, f"cache_idxs.{split}.csv"), "w") as csvfile:
            csvfile.write(
                    "\n".join(
                        [f"{idx},{length}" for idx, length in cache_idxs]
                        )
                    )
        cache_idxs = []
        cache = []


TABOO_LIST = []

def main():
    argp = ArgumentParser()
    argp.add_argument("ben_config", help="The model config")
    args = argp.parse_args()
    yaml_args = yaml.safe_load(open(args.ben_config))
    yaml_args['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # update sys.argv for OpenNMT run script
    cmdline_args = [sys.argv[0]]
    for key, val in yaml_args['onmt']['training'].items():
        if key in TABOO_LIST:
            continue
        cmdline_args.append(f"--{key}")
        if not isinstance(val, bool):
            cmdline_args.append(f"{val}")
    cmdline_args.append('--data')
    cmdline_args.append(yaml_args['onmt']['preprocessing']['data_dir'])
    sys.argv = cmdline_args

    onmt_parser = _get_parser()
    opt = onmt_parser.parse_args()

    save_hidden_states(opt, yaml_args)

if __name__ == "__main__":
    main()
