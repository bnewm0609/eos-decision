"""
Output some statistics about the dataset
"""
from collections import Counter
import numpy as np
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/translation/view_length_distribution.py [tsv flie]")
        sys.exit(0)

    filename = sys.argv[1]
    length_counter = Counter()
    with open(filename, "r") as tsv_file:
        for line in tsv_file:
            input_target = line.strip().split("\t")
            if len(input_target) != 2:
                continue
            _, target = input_target
            length_counter[len(target.split())] += 1

    total = sum(length_counter.values())
    print(f"Total: {total}")
    # output median, mean
    length_counter_lengths = []
    length_counter_counts = []

    for key in sorted(list(length_counter.keys())):
        length_counter_lengths.append(key)
        length_counter_counts.append(length_counter[key])


    mean = np.average(length_counter_lengths, weights=length_counter_counts)
    cuml_sum = np.cumsum(length_counter_counts)
    median_idx = np.where(cuml_sum < total / 2)[0][-1] + 1 # the bucket that the median falls into is 1 more than the last index that's after the median 
    median = length_counter_lengths[median_idx]

    # summarize
    print(f"Mean Length: {mean}")
    print(f"Median Length: {median}")
    print(f"Range: [{min(length_counter)}, {max(length_counter)}]")


if __name__ == "__main__":
    main()
