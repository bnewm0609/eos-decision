"""
Runs the Moses toknizatoin script for given source and target leanguages.
"""
import os
import subprocess
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python scripts/translation/moses_tokenize.py [tsv file] [src_lang] [tgt_lang]")
        sys.exit(0)

    tsv_filename = sys.argv[1]
    src_lang = sys.argv[2]
    tgt_lang = sys.argv[3]
    name, ext = os.path.splitext(tsv_filename)
    lang1_tokenized_filname = f"{name}_l1_tkn{ext}"
    lang2_tokenized_filname = f"{name}_l2_tkn{ext}"
    all_tokenized_filename = f"{name}_tokenized{ext}"
    subprocess.run(f"cat {tsv_filename} | cut -f1 | perl tools/tokenizer.perl -l {src_lang} > {lang1_tokenized_filname}", shell=True)
    subprocess.run(f"cat {tsv_filename} | cut -f2 | perl tools/tokenizer.perl -l {tgt_lang} > {lang2_tokenized_filname}", shell=True)
    subprocess.run(f"paste {lang1_tokenized_filname} {lang2_tokenized_filname} > {all_tokenized_filename}", shell=True)
    subprocess.run(f"rm {lang1_tokenized_filname}", shell=True) 
    subprocess.run(f"rm {lang2_tokenized_filname}", shell=True) 

if __name__ == "__main__":
    main()
