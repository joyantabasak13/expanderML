from schnetpack.data import ASEAtomsData
from schnetpack.datasets import OrganicMaterialsDatabase
from schnetpack.transform import ASENeighborList
import os
import schnetpack as spk
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
from schnetpack.data import AtomsLoader
omdb_dir = './omdb_dir'
if not os.path.exists('omdb_dir'):
    os.makedirs(omdb_dir)

OMDB_data = OrganicMaterialsDatabase(
    './omdb.db',
    batch_size=10,
    num_train=9000,
    num_val=1000,
    num_test= 2500,
    transforms=[
        trn.ASENeighborList(cutoff=5.)
    ],
    split_file=os.path.join(omdb_dir, "new_split.npz"),
    num_workers= 0,
    num_test_workers=0,
    num_val_workers=0,
    load_properties=[OrganicMaterialsDatabase.BandGap],
    property_units={OrganicMaterialsDatabase.BandGap: 'eV'},
    pin_memory=False
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

# print(OMDB_data.dataset.property_list['band_gap'])
cutoff = 5.
n_atom_basis = 64

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=25, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)
pred_BandGap= spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=OrganicMaterialsDatabase.BandGap)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_BandGap],
    postprocessors=[trn.CastTo64(), trn.AddOffsets(OrganicMaterialsDatabase.BandGap, add_atomrefs=True)]
)

output_BandGap = spk.task.ModelOutput(
    name=OrganicMaterialsDatabase.BandGap,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_BandGap],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(omdb_dir, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    default_root_dir=omdb_dir,
    max_epochs=2, # for testing, we restrict the number of epochs
)
converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)
inputs = converter(OMDB_data)
# print(OMDB_data.train_dataset())
# OMDB_data.test_dataloader()
# OMDB_data.val_dataloader()


# trainer.fit(task, datamodule=OMDB_data.train_dataloader(), OMDB_data.val_dataloader())
# trainer.fit(task, datamodule=OMDB_data)
# best_model = torch.load(os.path.join(omdb_dir, 'best_inference_model'))
#
# for batch in OMDB_data.test_dataloader():
#     result = best_model(batch)
#     print("Result dictionary:", result)
#     break
