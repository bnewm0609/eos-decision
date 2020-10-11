mkdir -p data/translation/de_en
wget -P data/translation/de_en http://stanford.edu/~blnewman/sample_data/de_en_sample.tsv.gz
gunzip data/translation/de_en/de_en_sample.tsv.gz
python scripts/translation/preprocess.py data/translation/de_en/de_en_sample.tsv --subsample 1000 --tokenize de,en --length-split 7 --dev-set --strip-punctuation
python scripts/translation/preprocess.py data/translation/de_en/subsampled_1000/de_en_sample_subsampled_tokenized_npunct.tsv --length-split 9 --dev-set
python scripts/translation/preprocess.py data/translation/de_en/subsampled_1000/de_en_sample_subsampled_tokenized_npunct.tsv --length-split 11 --dev-set

# Uncomment lines below for full dataset
#wget -P data/translation/de_en/ http://www.statmt.org/europarl/v9/training/europarl-v9.de-en.tsv.gz
#gunzip data/translation/de_en/europarl-v9.de-en.tsv.gz
#python scripts/translation/preprocess.py data/translation/de_en/europarl-v9.de-en.tsv --subsample 500000 --tokenize de,en --length-split 10 --dev-set --strip-punctuation
#python scripts/translation/preprocess.py data/translation/de_en/subsampled_500000/europarl-v9.de-en_subsampled_tokenized_npunct.tsv --length-split 15 --dev-set
#python scripts/translation/preprocess.py data/translation/de_en/subsampled_500000/europarl-v9.de-en_subsampled_tokenized_npunct.tsv --length-split 25 --dev-set

