"""
Separate languages to prepare for OpenNMT
"""
from argparse import ArgumentParser
import os
import subprocess


def main():
    argparser = ArgumentParser()
    argparser.add_argument("data_dir")
    argparser.add_argument("src_lang")
    argparser.add_argument("tgt_lang")
    args = argparser.parse_args()

    data_dir = args.data_dir
    filenames = ['dev', 'small_train', 'tasks_test_length']
    ext = 'txt'
    for filename in filenames:
        src_file = f"{filename}_{args.src_lang}.{ext}"
        src_file = os.path.join(data_dir, src_file)
        tgt_file = f"{filename}_{args.tgt_lang}.{ext}"
        tgt_file = os.path.join(data_dir, tgt_file)
        all_file = f"{filename}.{ext}"
        all_file = os.path.join(data_dir, all_file)

        subprocess.run(f"cut -f1 {all_file} > {src_file}", shell=True)
        subprocess.run(f"cut -f2 {all_file} > {tgt_file}", shell=True)

        # detokenize the dev and test target data as well:
        # if filename != 'small_train':
        #     tgt_dtkn_file = f"{filename}_{args.tgt_lang}.dtkn.{ext}"
        #     tgt_dtkn_file = os.path.join(data_dir, tgt_dtkn_file)
        #     subprocess.run(f"perl tools/detokenizer.perl -l {args.tgt_lang} < {tgt_file} > {tgt_dtkn_file}", shell=True)

if __name__ == "__main__":
    main()
