"""
Run a given script over all files in a directory (including files in subdirectories).
Given a white list and/or a black list, run the script over files with paths that match
or don't match strings int he white/black list respectively.

Related to ./run_all.py, but can be used for arbitrary scripts.
"""
from argparse import ArgumentParser
import os
import subprocess
import sys

FILE_PLACEHOLDER = "<PATH>"

def main():
    argp = ArgumentParser()
    argp.add_argument("script")
    argp.add_argument("base_dir")
    argp.add_argument("--white_list", help="Comma separated list of strings. Only run configs with a string in their path.")
    argp.add_argument("--black_list", help="Comma separated list of strings. Don't run configs with these strings in their path (overrides any strings in --white_list if present).")
    args = argp.parse_args()

    black_list = args.black_list.split(",") if args.black_list else None
    white_list = args.white_list.split(",") if args.white_list else None

    if not os.path.isdir(args.base_dir):
        print(f"{args.base_dir} is not a directory")
        sys.exit(1)

    for dir_name, sub_dirs, file_list in os.walk(args.base_dir):
        for config in file_list:
            path = os.path.join(dir_name, config)
            if black_list and any([item in path for item in black_list]):
                continue
            if white_list and not any([item in path for item in white_list]):
                continue
            script_args = args.script.split()
            if FILE_PLACEHOLDER in script_args:
                script_args = [arg if arg != FILE_PLACEHOLDER else path for arg in script_args]
            else:
                script_args.append(path)

            print(f"> python {' '.join(script_args)}")
            subprocess.run(["python", *script_args])


if __name__ == "__main__":
    main()
