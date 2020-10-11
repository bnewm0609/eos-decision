"""
Removes the EOS token (and other ending tokens) from the end of Dyck examples.
NOTE: Outputs to standard out.
"""
from argparse import ArgumentParser
import numpy as np

def main():
    argp = ArgumentParser()
    argp.add_argument("basefile", help="File containing the Dyck data")
    args = argp.parse_args()

    with open(args.basefile) as df:
        for line in df:
            tkns = line.strip().split()
            tkns = tkns[: -1]
            print(" ".join(tkns))

if __name__ == "__main__":
    main()
