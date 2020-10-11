"""
Create a length split from Europarl data.
"""
import os
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/gen_tsv_mt_length_splits.py <mt_file.tsv> <len_cutoff>")
        sys.exit()

    # read in args from command line
    tsv_filename = sys.argv[1]
    data_dir = os.path.dirname(tsv_filename)
    length_cutoff = int(sys.argv[2])

    # first read in the tokenized tsv file
    long_data, short_data = [], []
    with open(tsv_filename) as tsvf:
        for line in tsvf:
            input_target = line.strip().split("\t")
            if len(input_target) != 2:
                continue
            src_ex, tgt_ex = input_target

            # then filter by **target** length
            if len(tgt_ex.split()) > length_cutoff:
               long_data.append(line) 
            else:
                short_data.append(line)

    print(f"Number of short data samples: {len(short_data)}")
    print(f"Number of long data samples: {len(long_data)}")
    # finally, save the tokenized files back into tsv format
    datasplit_path = os.path.join(data_dir, f"length_{length_cutoff}_split")
    os.makedirs(datasplit_path)
    long_filename = os.path.join(datasplit_path, "tasks_test_length.txt")
    short_filename = os.path.join(datasplit_path, "tasks_train_length.txt")

    with open(long_filename, "w") as f:
        f.write("".join(long_data))

    with open(short_filename, "w") as f:
        f.write("".join(short_data))

if __name__ == "__main__":
    main()
