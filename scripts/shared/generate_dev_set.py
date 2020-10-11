"""
Generates the development set given the training data.

Chooses min(VAL_PERCENT, MAX_VAL_SIZE) random lines and separates them into 
a separate file. Assumes that each line is a sepate training example.
"""
import os
import random
import sys

random.seed(425)

VAL_PERCENT = 0.05
MAX_VAL_SIZE = 1100


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_dev_set.py [train_data.txt]")

    data_filename = sys.argv[1]
    data_dir = os.path.dirname(data_filename)

    # Separate train and dev data
    new_train_data = []
    val_data = []
    with open(data_filename, 'r') as train_data:
        train_data = train_data.readlines()
        train_size = len(train_data)
        val_size = min(int(train_size * VAL_PERCENT), MAX_VAL_SIZE)
        print(f"Old Train Size: {train_size}")
        print(f"New Train Size: {train_size - val_size}")
        print(f"Val Size: {val_size}")
        val_idxs = set(random.sample(range(train_size), val_size))

        for i in range(train_size):
            if i in val_idxs:
                val_data.append(train_data[i])
            else:
                new_train_data.append(train_data[i])


    new_train_filename = f"{data_dir}/small_train.txt"
    val_filename = f"{data_dir}/dev.txt"
    
    # Save train data
    f = open(new_train_filename, 'w')
    f.write(''.join(new_train_data))
    f.close()

    # Save dev data
    f = open(val_filename, 'w')
    f.write(''.join(val_data))
    f.close()

if __name__ == "__main__":
    main()
