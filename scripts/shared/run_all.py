"""
Runs all of the experiments from configs in a given directory. A mode can be
specified indicating how the experiment should be run (e.g. what training
procedure to use or if sequences should be generated or models evaluated).
A black/white_list can be specified to only run configs that contain (or do not)
contain a certain substring.
"""
from argparse import ArgumentParser
import os
import subprocess

TRAIN_SEQ2SEQ_MODE = 'train-seq2seq'
TRAIN_TRUNCATION_MODE = 'train-truncation'
EVAL_MODE = 'eval'
TRAIN_ONMT_MODE = 'train-onmt'
GEN_ONMT_MODE = 'generate-onmt'
VALID_MODES = (TRAIN_SEQ2SEQ_MODE, TRAIN_TRUNCATION_MODE, EVAL_MODE, TRAIN_ONMT_MODE, GEN_ONMT_MODE)

def main():
    argp = ArgumentParser()
    argp.add_argument("base_dir", help="The root directory under which all configs will be run.")
    argp.add_argument("mode", help="The mode to run the configs with. One of `train`, `generate`, `eval`, `train-truncation`.", choices=VALID_MODES)
    argp.add_argument("--white_list", help="Comma separated list of strings. Only run configs with a string in their path.")
    argp.add_argument("--black_list", help="Comma separated list of strings. Don't run configs with these strings in their path (overrides any strings in --white_list if present).")
    args = argp.parse_args()

    black_list = args.black_list.split(",") if args.black_list else None
    white_list = args.white_list.split(",") if args.white_list else None

    for dir_name, sub_dirs, file_list in os.walk(args.base_dir):
        for config in file_list:
            path = os.path.join(dir_name, config)

            if black_list and any([item in path for item in black_list]):
                continue
            if white_list and not any([item in path for item in white_list]):
                continue

            if 'onmt' in args.mode:
                cmd = f"python scripts/translation/onmt_preprocess.py {path}"
                if args.mode == TRAIN_ONMT_MODE:
                    cmd += " --train-seq2seq"
            else:
                cmd = f"python src/run.py {path}"
                if args.mode == TRAIN_TRUNCATION_MODE:
                    cmd += " --train-truncation"
                elif args.mode == TRAIN_SEQ2SEQ_MODE:
                    cmd += " --train-seq2seq"

            print(f"> {cmd}")
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
