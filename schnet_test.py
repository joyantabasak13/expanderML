import os
import schnetpack as spk
from schnetpack.datasets import OrganicMaterialsDatabase as OMDB
from torch.optim import Adam
import schnetpack.train as trn
import torch
import numpy as np
import datetime
from torch.utils.data.sampler import RandomSampler
import math

ct = datetime.datetime.now()
print("current time:-", ct)

omdbase = './omdbase'
if not os.path.exists('omdbase'):
    os.makedirs(omdbase)

omdata = OMDB('/Users/joyanta/Documents/Research/Materials/OMDB_dataset/OMDB-GAP1_v1.1.tar.gz')

train, val, test = spk.train_test_split(
    data=omdata,
    num_train=9000,
    num_val=1000,
    split_file=os.path.join(omdbase, "split.npz"),
)

batchsize = 32
n_epochs = 50
lr_rate = 1e-3

train_loader = spk.AtomsLoader(train, batch_size=batchsize, sampler=RandomSampler(train), pin_memory=True)
val_loader = spk.AtomsLoader(val, batch_size=batchsize, pin_memory=True)

schnet = spk.representation.SchNet(
    n_atom_basis=64,
    n_filters=64,
    n_gaussians=25,
    n_interactions=3,
    cutoff=5.,
    cutoff_network=spk.nn.cutoff.CosineCutoff,
    normalize_filter=True
)

# atomrefs = omdata.train_dataset.atomrefs
# means, stddevs = omdata.get_stats()

output_gap = spk.atomistic.Atomwise(n_in=64, property=OMDB.BandGap)

model = spk.AtomisticModel(representation=schnet, output_modules=output_gap)

optimizer = Adam(model.parameters(), lr=lr_rate)

print("After Initializer")
# os.system("rm -r ./omdbase/checkpoints")
# os.system("rm -r ./omdbase/log.csv")


loss = trn.build_mse_loss([OMDB.BandGap])
metrics = [spk.metrics.MeanAbsoluteError(OMDB.BandGap)]
hooks = [
    trn.CSVHook(log_path=omdbase, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=10, factor=0.6, min_lr=1e-4,window_length=1,
        stop_after_min=True
    )
]


trainer = trn.Trainer(
    model_path=omdbase,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

print("After Model Declaration")

device = "cpu"  # change to 'cpu' if gpu is not available

ct = datetime.datetime.now()
print("current time:-", ct)

trainer.train(device=device, n_epochs=n_epochs)

print("After Training")

ct = datetime.datetime.now()
print("current time:-", ct)

best_model = torch.load(os.path.join(omdbase, 'best_model'))

test_loader = spk.AtomsLoader(test, batch_size=32)

err = 0
err_mse = 0
print(len(test_loader))
for count, batch in enumerate(test_loader):
    # move batch to GPU, if necessary
    batch = {k: v.to(device) for k, v in batch.items()}

    # apply model
    pred = best_model(batch)

    # calculate absolute error
    tmp = torch.sum(torch.abs(pred[OMDB.BandGap] - batch[OMDB.BandGap]))
    tmp = tmp.detach().cpu().numpy()  # detach from graph & convert to numpy
    err += tmp

    tmp = torch.sum(pow(torch.abs(pred[OMDB.BandGap] - batch[OMDB.BandGap]), 2))
    tmp = tmp.detach().cpu().numpy()  # detach from graph & convert to numpy
    err_mse += tmp

    # log progress
    percent = '{:3.2f}'.format(count / len(test_loader) * 100)
    print('Progress:', percent + '%' + ' ' * (5 - len(percent)))

print("TestLoader Finished")
ct = datetime.datetime.now()
print("current time:-", ct)
err /= len(test)
err_mse /= len(test)
print('Test MAE ', np.round(err, 2))
print('Test MSE ', np.round(err_mse, 2))
print('Test RMSE ', math.sqrt(np.round(err_mse, 2)))
