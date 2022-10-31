import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn
import torch
import numpy as np
from ase import Atoms

import torch
import torchmetrics
import pytorch_lightning as pl

qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)

# %rm new_split.npz

qm9data = QM9(
    './qm9.db',
    batch_size=100,
    num_train=1000,
    num_val=1000,
    # transforms=[
    #     trn.ASENeighborList(cutoff=5.),
    #     trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
    #     trn.CastTo32()
    # ],
    property_units={QM9.U0: 'eV'},
    num_workers=0,
    split_file=os.path.join(qm9tut, "new_split.npz"),
    pin_memory=False, # set to false, when not using a GPU
    load_properties=[QM9.U0], #only load U0 property
)
qm9data.prepare_data()
qm9data.setup()

atomrefs = qm9data.train_dataset.atomrefs
print(atomrefs)
print('U0 of hyrogen:', atomrefs[QM9.U0][1].item(), 'eV')
print('U0 of carbon:', atomrefs[QM9.U0][6].item(), 'eV')
print('U0 of oxygen:', atomrefs[QM9.U0][8].item(), 'eV')
means, stddevs = qm9data.get_stats(
    QM9.U0, divide_by_atoms=True, remove_atomref=True
)
print('Mean atomization energy / atom:', means.item())
print('Std. dev. atomization energy / atom:', stddevs.item())

cutoff = 5.
n_atom_basis = 30

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)
pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_U0],
    postprocessors=[trn.CastTo64(), trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)]
)
output_U0 = spk.task.ModelOutput(
    name=QM9.U0,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_U0],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

# logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(qm9tut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    default_root_dir=qm9tut,
    max_epochs=3, # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=qm9data)


best_model = torch.load(os.path.join(qm9tut, 'best_inference_model'))

for batch in qm9data.test_dataloader():
    result = best_model(batch)
    print("Result dictionary:", result)
    break

converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32)

numbers = np.array([6, 1, 1, 1, 1])
positions = np.array([[-0.0126981359, 1.0858041578, 0.0080009958],
                      [0.002150416, -0.0060313176, 0.0019761204],
                      [1.0117308433, 1.4637511618, 0.0002765748],
                      [-0.540815069, 1.4475266138, -0.8766437152],
                      [-0.5238136345, 1.4379326443, 0.9063972942]])
atoms = Atoms(numbers=numbers, positions=positions)

inputs = converter(atoms)

print('Keys:', list(inputs.keys()))

pred = best_model(inputs)

print('Prediction:', pred[QM9.U0])

calculator = spk.interfaces.SpkCalculator(
    model_file=os.path.join(qm9tut, "best_inference_model"), # path to model
    neighbor_list=trn.ASENeighborList(cutoff=5.), # neighbor list
    energy_key=QM9.U0, # name of energy property in model
    energy_unit="eV", # units of energy property
    device="cpu", # device for computation
)
atoms.set_calculator(calculator)
print('Prediction:', atoms.get_total_energy())