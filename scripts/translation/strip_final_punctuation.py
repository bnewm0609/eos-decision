"""
Removes last token from end of line if it's a punctuation mark while maintaining
tab-separation if peresent.
"""
from argparse import ArgumentParser
import string

def main():
    argp = ArgumentParser()
    argp.add_argument('data_file')
    args = argp.parse_args()

    with open(args.data_file, "r") as dfile:
        for line in dfile:
            all_tkns = line.strip().split("\t")
            if len(all_tkns[-1]) == 0:
                continue
            tkns = all_tkns[-1].split()
            if tkns[-1] in string.punctuation:
                tkns = tkns[:-1]
            result = " ".join(tkns)
            all_tkns[-1] = result
            print("\t".join(all_tkns))

if __name__ == "__main__":
    main()
