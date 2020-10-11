# Appending to sys.path is bad style but we do it to access our src files
import sys; sys.path.append("src")

from argparse import ArgumentParser
import numpy as np
import os

import utils


VALID_METRICS = {"ppl", "close_bracket", "probe_acc"}


def get_ppl(path):
    summary = open(path).readlines()[-1].strip()
    return float(summary)

def get_mean(path):
    try:
        summary = open(path).readlines()[:-1]
    except FileNotFoundError:
        raise FileNotFoundError
    _, accs = zip(*[(x.split(",")) for x in summary])
    accs = list(map(float, accs))
    mean = np.mean(accs)
    return mean

    return float(summary.split()[0])

metric_file = "close_bracket_acc.{}"
get_metric = get_mean


def main():
    argp = ArgumentParser()
    argp.add_argument("base_dir")
    argp.add_argument("--white_list")
    argp.add_argument("--black_list")
    argp.add_argument("--metric", choices=VALID_METRICS)
    cli_args = argp.parse_args()

    black_list = cli_args.black_list.split(",") if cli_args.black_list else None
    white_list = cli_args.white_list.split(",") if cli_args.white_list else None
    if cli_args.metric is None or cli_args.metric == "close_bracket":
        get_metric = get_mean
        metric_file = "close_bracket_acc.{}"
    elif cli_args.metric == "ppl":
        get_metric = get_ppl
        metric_file = "{}.perplexity"
    elif cli_args.metric == "probe_acc":
        get_metric = get_ppl
        metric_file = "dyck_ending_acc.{}"

    dev_medians = []
    test_medians = []
    print("Iterating through file system")
    for config in os.listdir(cli_args.base_dir):
        if config.startswith("."):
            continue
        if black_list and any([item in config for item in black_list]):
            continue
        if white_list and not any([item in config for item in white_list]):
            continue

        print(f"\t{config}")
        config_path = os.path.join(cli_args.base_dir, config)
        try:
            config_args = utils.load_config_from_path(config_path)
            results_dir = utils.get_results_dir_of_args(config_args)
            dev_results_path = os.path.join(results_dir, metric_file.format("dev"))
            dev_medians.append(get_metric(dev_results_path))
            test_results_path = os.path.join(results_dir, metric_file.format("test"))
            test_medians.append(get_metric(test_results_path))
        except FileNotFoundError:
            print(f"Not found: {config_path}")
            continue

    dev_median = np.median(dev_medians)
    test_median = np.median(test_medians)
    print(f"In domain median: {dev_median} (out of {len(dev_medians)})")
    print(f"Out of domain median: {test_median} (out of {len(test_medians)})")


if __name__ == "__main__":
    main()
