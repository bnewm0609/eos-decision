# Dyck-2,4
mkdir -p data/dyck_mk/dyck_4_2/no_eos/
mkdir -p data/dyck_mk/dyck_4_2/eos/
python src/generate_mbounded_dyck.py configs/dyck_km/generate_data/dyck24.yaml
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_4_2/eos/train.formal.txt > data/dyck_mk/dyck_4_2/no_eos/train.formal.txt
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_4_2/eos/dev.formal.txt > data/dyck_mk/dyck_4_2/no_eos/dev.formal.txt
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_4_2/eos/test.formal.txt > data/dyck_mk/dyck_4_2/no_eos/test.formal.txt


# Dyck-2,6
mkdir -p data/dyck_mk/dyck_6_2/no_eos/
mkdir -p data/dyck_mk/dyck_6_2/eos/
python src/generate_mbounded_dyck.py configs/dyck_km/generate_data/dyck26.yaml
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_6_2/eos/train.formal.txt > data/dyck_mk/dyck_6_2/no_eos/train.formal.txt
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_6_2/eos/dev.formal.txt > data/dyck_mk/dyck_6_2/no_eos/dev.formal.txt
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_6_2/eos/test.formal.txt > data/dyck_mk/dyck_6_2/no_eos/test.formal.txt


# Dyck-2,8
mkdir -p data/dyck_mk/dyck_8_2/no_eos/
mkdir -p data/dyck_mk/dyck_8_2/eos/
python src/generate_mbounded_dyck.py configs/dyck_km/generate_data/dyck28.yaml
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_8_2/eos/train.formal.txt > data/dyck_mk/dyck_8_2/no_eos/train.formal.txt
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_8_2/eos/dev.formal.txt > data/dyck_mk/dyck_8_2/no_eos/dev.formal.txt
python scripts/dyck_km/remove_eos.py data/dyck_mk/dyck_8_2/eos/test.formal.txt > data/dyck_mk/dyck_8_2/no_eos/test.formal.txt
