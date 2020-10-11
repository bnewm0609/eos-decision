"""
Generate configs with hyperparameters.
(All configs should already be created in this repo).
"""
from argparse import ArgumentParser
import copy
import itertools
import os
import re
import yaml

# This can be expanded as needed
subs = [
        (",", "+"),
        ("lm/", ""),
        ("truncation/", "t"),
        ("seed", "Sd"),
        ("embed_dim", "Ed"),
        ("hidden_dim", "Hd"),
        ("False", "F"),
        ("True", "T"),
        ("use_eos", "ue"),
        ("eos_name", "en"),
        ("batch_size", "bs"),
        ]


def substitute(config, sub_phrase, value):
    """Recursively iterate through config substituting entries with 'sub_phrase' with ''value'"""
    sub_phrase_brackets = f"{{{sub_phrase}}}"
    # recurse  through all values
    # substitute in value whenever we see {subphrase}

    # we have more keys
    if isinstance(config, dict):
        for key in config:
            config[key] = substitute(config[key], sub_phrase, value)
    # base cases
    elif isinstance(config, list):
        for i in range(len(config)):
            config[i] = re.sub(sub_phrase_brackets, str(value), str(config[i]))
    elif isinstance(config, str):
        config = re.sub(sub_phrase_brackets, str(value), str(config))

    return config

def _set_config(config, path, value):
    edges = path.split("/")
    for edge in edges[:-1]:
        config = config.get(edge)
        if config is None:
            return False
    if edges[-1] not in config:
        return False
    config[edges[-1]] = value
    return True

def update_config(config, parameters):
    for path, value in parameters:
        paths = path.split(",") # account for multiple paths

        # account for having multiple values linked together
        if not isinstance(value, list):
            values = [value]
        else:
            values = value

        assert len(values) == len(paths) or len(values) == 1
        for path, value in itertools.zip_longest(paths, values, fillvalue=values[0]):
            # print(path, value)
            success = _set_config(config, path, value)
            if not success:
                config = substitute(config, path, value)

def experiment_name(parameters):
    """Creates a filename from parameter paths and values"""
    filename = []
    for path, value in parameters:
        update_value = False
        if isinstance(value, list):
            value = "+".join([str(v) for v in value])
            update_value = True
        for before, after in subs:
            path = re.sub(before, after, path)
            if update_value:
                value = re.sub(before, after, value)

        filename.append("-".join([path, str(value)]))
    return f'{"_".join(filename)}.yaml'



def main():
    argp = ArgumentParser()
    argp.add_argument("base_config", help="The config with all the default args")
    argp.add_argument("hyperparams", help="Paths to hyperparameters")
    argp.add_argument("--save_dir", help="Directory to save configs")
    args = argp.parse_args()

    # set up directory to save to
    save_dir = args.save_dir if args.save_dir is not None else "."
    os.makedirs(save_dir, exist_ok=True)

    # open the base config
    base_config = yaml.safe_load(open(args.base_config))

    # open the hyperparameters
    hyperparams = yaml.safe_load(open(args.hyperparams))
    paths, params = zip(*hyperparams.items())
    hyperparams = []
    for path, param_list in zip(paths, params):
        path_hyperparams = []
        for param in param_list:
            path_hyperparams.append((path, param))
        hyperparams.append(path_hyperparams)

    for param_set in itertools.product(*hyperparams):
        config_copy = copy.deepcopy(base_config)
        update_config(config_copy, param_set) # updates config in place
        config_name = experiment_name(param_set)
        save_path = os.path.join(save_dir, config_name)
        yaml.dump(config_copy, open(save_path, "w"))
        print(save_path)

if __name__ == "__main__":
    main()
