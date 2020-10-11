"""
Subsamples k lines from Europarl (tab-separated) dataset
"""
from argparse import ArgumentParser
import random
import os

random.seed(200)

# use resevoir sampling to sample k lines
def sample_from_file(filename, k):
    lines = []
    with open(filename, "r") as datafile:
        for i, line in enumerate(datafile):
            if len(lines) < k:
                lines.append(line)
            else:
                replace_idx = random.randrange(i)
                if replace_idx < k:
                    lines[replace_idx] = line
    return lines

def main():
    argp = ArgumentParser()
    argp.add_argument("data_file", type=str)
    argp.add_argument("k", type=int)
    argp_args = argp.parse_args()
    k = argp_args.k
    filename = argp_args.data_file

    sampled_lines = sample_from_file(filename, k)
    new_dir_name = f'subsampled_{k}'
    path = os.path.dirname(filename)
    new_dir_path = os.path.join(path, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    name, ext = os.path.splitext(os.path.basename(filename))
    name = os.path.join(f"{new_dir_path}", f"{name}_subsampled{ext}")
    with open(name, "w") as outfile:
        outfile.write("".join(sampled_lines))


if __name__ == "__main__":
    main()
