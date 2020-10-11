from argparse import ArgumentParser

def calculate_max_depth(sequence):
    state = []
    max_depth = 0
    for token in sequence.split():
        if ")" in token:
            state = state[:-1]
        else:
            state.append(token[0])
            max_depth = max(max_depth, len(state))

    return max_depth

def main():
    argp = ArgumentParser()
    argp.add_argument("data")
    args = argp.parse_args()

    min_length = float("inf")
    max_length = 0
    with open(args.data) as f:
        line = next(f)
        num_examples = 1
        max_depth = calculate_max_depth(line)
        eos = line.split()[-1] == "END"
        for line in f:
            max_length = max(max_length, len(line.split()))
            min_length = min(min_length, len(line.split()))
            num_examples += 1
        
    print(f"Maximum Depth: {max_depth}")
    print(f"Min Length: {min_length}")
    print(f"Max Length: {max_length}")
    print(f"Num Examples: {num_examples}")
    print(f"EOS? : {eos}")

if __name__ == "__main__":
    main()

