"""
preprocess.py
----

Preprocesses the translation data. Run from project home directory.
"""
from argparse import ArgumentParser
import os
import subprocess

# subsampler
SUBSAMPLER_CMD = "python scripts/translation/subsample.py {} {}"
SUBSAMPLE_DIR = "subsampled_{}"
SUBSAMPLE_DATA_FILE =  "{}_subsampled{}"

# tokenizer
TOKENIZER_CMD = "python scripts/translation/moses_tokenize.py {} {} {}"
TOKENIZER_DATA_FILE = "{}_tokenized{}"

# strip punctuation
STRIP_PUNCT_CMD = "python scripts/translation/strip_final_punctuation.py {} > {}"
STRIP_PUCNT_DATA_FILE = "{}_npunct{}"

# length splits
LENGTH_SPLIT_CMD = "python scripts/translation/generate_length_splits.py {} {}"
LENGTH_SPLIT_DIR = "length_{}_split"
LENGTH_SPLIT_DATA_FILE = "tasks_train_length.txt"
LENGTH_SPLIT_TEST_DATA_FILE = "tasks_test_length.txt"

# dev data generation
DEV_DATA_CMD = "python scripts/shared/generate_dev_set.py {}"
DEV_DATA_FILE = "dev.txt" # unused
SMALL_TRAINING_DATA_FILE = "small_train.txt" # unused

def parse_cli():
    argp = ArgumentParser()
    argp.add_argument("data_file", help="Path to data file to preprocess")
    argp.add_argument("--subsample", default=-1, help="Number of lines to subsample from the data file", type=int)
    argp.add_argument("--tokenize", default=None, help="Tokenize the lines using these comma-separated languages")
    argp.add_argument("--length-split", default=-1, help="Split data between train and test sets, with this argument being the maximum length of training targets", type=int)
    argp.add_argument("--dev-set", action='store_true', help="Set this flag to generate a dev set")
    argp.add_argument("--strip-punctuation", action='store_true', help="Remove final punctuation from target sequences")
    return argp.parse_args()


def subsample(data_file, k):
    """
    Creates directory at same level as `data_file` with name
    SUBSAMPLE_DIR and a data file called SUBSAMPLE_DATA_FILE
    """
    cmd = SUBSAMPLER_CMD.format(data_file, k)
    subprocess.run(cmd, shell=True, check=True)

def tokenize(data_file, src_lang_id, tgt_lang_id):
    """
    Creates data file with name TOKENIZED_DATA_FILE
    """
    cmd = TOKENIZER_CMD.format(data_file, src_lang_id, tgt_lang_id)
    subprocess.run(cmd, shell=True, check=True)

def gen_length_splits(data_file, ls):
    """
    Creates a directory with name LENGTH_SPLITS_DIR
    """
    cmd = LENGTH_SPLIT_CMD.format(data_file, ls)
    subprocess.run(cmd, shell=True, check=True)

def gen_dev_set(training_data_file, length_split_dir):
    """
    Creates files with name DEV_DATA_FILE and SMALL_TRAIN_DATA_FILE
    """
    cmd = DEV_DATA_CMD.format(training_data_file)
    subprocess.run(cmd, shell=True, check=True)
    old_training_file = os.path.join(length_split_dir, LENGTH_SPLIT_DATA_FILE)
    cmd = f"rm {old_training_file}"
    subprocess.run(cmd, shell=True, check=True)

def strip_punctuation(data_file, out_file):
    cmd = STRIP_PUNCT_CMD.format(data_file, out_file)
    subprocess.run(cmd, shell=True, check=True)
    # uncommment below if you want to remove the old files automatically
    # cmd = f"rm {data_file}"
    # subprocess.run(cmd, shell=True, check=True)

def main():
    """
    Determine which preprocessing to do
    """
    args = parse_cli()

    data_file = args.data_file
    data_dir = os.path.dirname(args.data_file)

    if args.subsample > 0:
        subsample(data_file, args.subsample)
        # update data_dir and data_file for the next script
        basename, ext = os.path.splitext(os.path.basename(data_file))
        data_dir = os.path.join(data_dir, SUBSAMPLE_DIR.format(args.subsample))
        data_file = os.path.join(data_dir, SUBSAMPLE_DATA_FILE.format(basename, ext))
        print(f"Subsampling produced: {data_file}")
        assert os.path.exists(data_file), "FAILURE: file does not exist"

    if args.tokenize is not None:
        src_lang, tgt_lang = args.tokenize.split(",")
        tokenize(data_file, src_lang, tgt_lang)
        basename, ext = os.path.splitext(os.path.basename(data_file))
        data_file = os.path.join(data_dir, TOKENIZER_DATA_FILE.format(basename, ext))
        print(f"Tokenizer is creating: {data_file}")
        assert os.path.exists(data_file), "FAILURE: file does not exist"

    if args.strip_punctuation:
        basename, ext = os.path.splitext(os.path.basename(data_file))
        out_file = os.path.join(data_dir, STRIP_PUCNT_DATA_FILE.format(basename, ext))
        strip_punctuation(data_file, out_file)
        data_file = out_file
        print(f"Stripping Punctuation is creating: {data_file}")
        assert os.path.exists(data_file), "FAILURE: file does not exist"

    if args.length_split > 0:
        gen_length_splits(data_file, args.length_split)
        data_dir = os.path.join(data_dir, LENGTH_SPLIT_DIR.format(args.length_split))
        data_file = os.path.join(data_dir, LENGTH_SPLIT_DATA_FILE)
        print(f"Length split tried to produce: {data_dir}")
        assert os.path.exists(data_dir), "FAILURE: directory not found"

    if args.dev_set:
        gen_dev_set(data_file, data_dir)




if __name__ == "__main__":
    main()
