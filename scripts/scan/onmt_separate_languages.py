"""
Sepeartes the SCAN input and output strings into different files as required by
OpenNMT.
"""
from argparse import ArgumentParser
import os

def main():
    argparser = ArgumentParser()
    argparser.add_argument("data_dir")
    argparser.add_argument("output_dir", nargs="?")
    args = argparser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir if args.output_dir else args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    filenames = list(os.listdir(data_dir))
    for filename in filenames:
        filepath = os.path.join(data_dir, filename)
        if os.path.isdir(filepath): continue
        outfile_src = f"{os.path.join(out_dir, filename)}.src"
        outfile_tgt = f"{os.path.join(out_dir, filename)}.tgt"
        srcs = []
        tgts = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                src, tgt = line.split(" OUT: ")
                src = " ".join(src.split()[1:])
                tgt = " ".join(tgt.split())
                srcs.append(src)
                tgts.append(tgt)

        with open(outfile_src, "w") as f:
            f.write("\n".join(srcs))

        with open(outfile_tgt, "w") as f:
            f.write("\n".join(tgts))

if __name__ == "__main__":
    main()
