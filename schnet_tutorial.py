from schnetpack.data import ASEAtomsData
from schnetpack.datasets import OrganicMaterialsDatabase
from schnetpack.transform import ASENeighborList
import os
import schnetpack as spk
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl

OMDB_data = OrganicMaterialsDatabase(
    './omdb.db',
    batch_size=10,
    num_train=9000,
    num_val=1000,
    num_test= 2500,
    transforms=[ASENeighborList(cutoff=5.)],
    split_file="new_split_omdb.npz",
    num_workers= 0,
    num_test_workers=0,
    num_val_workers=0
)
OMDB_data.prepare_data()
OMDB_data.setup()

print('Number of reference calculations:', len(OMDB_data.dataset))
print('Number of train data:', len(OMDB_data.train_dataset))
print('Number of validation data:', len(OMDB_data.val_dataset))
print('Number of test data:', len(OMDB_data.test_dataset))
print('Available properties:')

for p in OMDB_data.dataset.available_properties:
    print('-', p)


