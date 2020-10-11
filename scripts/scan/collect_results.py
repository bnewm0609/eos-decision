"""
The SCAN outputs are in a human-readable format, so this file parses them, 
computes some summary statistics, and outputs a latex table.
"""
# Appending to sys.path is bad style but we do it to access our src files
import sys; sys.path.append("src")

from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import re
import os

import utils

seed_pattern_lstm = re.compile("ls-(\d\d)_(.*)\/config_seed(?:50\d)")
seed_pattern_transformer = re.compile("ls-(\d\d)_(?:.*)-(.*).yaml")
# seed_pattern_2 = re.compile("ls-(\d\d)_.*+(.*)+(?)")
# seed_pattern_2 = re.compile("ls(\d\d)_(.*)_seed(50\d)")

def get_result(path):
    try:
        summary = open(path).readlines()[-1].strip()
        return float(summary)
    except:
        return None

def main():
    argp = ArgumentParser()
    argp.add_argument("base_dir")
    cli_args = argp.parse_args()

    test_results = defaultdict(list)
    results = []
    print("Iterating through file system")
    # for config in os.listdir(cli_args.base_dir):
    for dir_name, sub_dirs, file_list in os.walk(cli_args.base_dir):
        for config in file_list:
            config = os.path.join(dir_name, config)
            # import ipdb; ipdb.set_trace()

            for pattern in (seed_pattern_lstm, seed_pattern_transformer):
                s = re.findall(pattern, config)
                if s:
                    break
            if not s: # no match
                continue

            ls, eos  = s[0]
            ls = int(ls)
            print(f"\t{config}")
            # config_path = os.path.join(cli_args.base_dir, config)
            # import ipdb; ipdb.set_trace()
            config_args = utils.load_config_from_path(config) #_path)
            results_dir = utils.get_results_dir_of_args(config_args)
            dev_results_path = os.path.join(results_dir, "oracle_exact_match_acc.dev_sampled")
            dev_result = get_result(dev_results_path)
            if dev_result is None:
                print("\t> error with dev - check if file exists")
                continue
            test_results_path = os.path.join(results_dir, "oracle_exact_match_acc.test_sampled")
            test_result = get_result(test_results_path)
            if test_result is None:
                print("\t> error with test - check if file exists")
                continue
            results.append((config, dev_result, test_result))
            test_results[(eos, ls)].append(test_result)

    out_str = ""
    for k in sorted(test_results.keys()):
        out_str += " & {:.2f}".format(np.median(test_results[k]))
        if k[1] == 40: # max length split
            out_str += "\\\\\n"
    print(out_str)


if __name__ == "__main__":
    main()
