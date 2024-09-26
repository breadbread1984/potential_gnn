#!/bin/bash

mkdir -P trainset
mkdir -P evalset
python3 create_dataset.py --smiles CC --bond_dist 0.6 --output trainset &
python3 create_dataset.py --smiles CC --bond_dist 0.7 --output evalset &
python3 create_dataset.py --smiles CC --bond_dist 0.8 --output trainset &
python3 create_dataset.py --smiles CC --bond_dist 0.9 --output evalset &
python3 create_dataset.py --smiles CC --bond_dist 1.0 --output trainset &
python3 create_dataset.py --smiles CC --bond_dist 1.1 --output evalset &
python3 create_dataset.py --smiles CC --bond_dist 1.2 --output trainset &
python3 create_dataset.py --smiles CC --bond_dist 1.3 --output evalset &
python3 create_dataset.py --smiles CC --bond_dist 1.4 --output trainset &
python3 create_dataset.py --smiles CC --bond_dist 1.5 --output evalset &
python3 create_dataset.py --smiles C --bond_dist 0.6 --output trainset &
python3 create_dataset.py --smiles C --bond_dist 0.7 --output evalset &
python3 create_dataset.py --smiles C --bond_dist 0.8 --output trainset &
python3 create_dataset.py --smiles C --bond_dist 0.9 --output evalset &
python3 create_dataset.py --smiles C --bond_dist 1.0 --output trainset &
python3 create_dataset.py --smiles C --bond_dist 1.1 --output evalset &
python3 create_dataset.py --smiles C --bond_dist 1.2 --output trainset &
python3 create_dataset.py --smiles C --bond_dist 1.3 --output evalset &
python3 create_dataset.py --smiles C --bond_dist 1.4 --output trainset &
python3 create_dataset.py --smiles C --bond_dist 1.5 --output evalset &
wait
