"""
Contains training regimen for truncation models---ones that predict where
sequences end. Right now this is only the probe for the Dyck-k,m languages.

The main difference between this and regimen.py is that the truncation
models calculate their own losses.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils


EARLY_STOPPING_THRESHOLD = 4
MAX_EPOCHS = 5000
MIN_DEV_LOSS_THRESHOLD = 0.0001

def run_truncation_batches(data, tm, optimizer, name, pad_id, epoch_idx):
    """
    Runs the model over an epoch of training/development data to calculate loss.

    Args:
        data (torch.utils.data.Dataloader): the train/dev data iterable
        tm (models.Seq2Seq): the truncation model we're calculate the loss of
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Module): the loss function
        name (str): label for the batch: either "train" or "dev"
        pad_id (int): the index of the padding token in the data
        epoch_idx (int): the epoch number. Used for noam opt, but really
            shouldn't be.
    Returns:
        tuple(float, int) with total loss (aggregated accross batches) and the
        total number of batches.
    """
    total_loss = 0
    num_batches = 0
    training = name == "train"
    if training:
        tm.train()
    else:
        tm.eval()
    for (
            batch_i,
            (inputs_batch,
            (targets_batch_in, targets_batch_out),
            (input_length_batch, target_length_batch),
            _)
        ) in tqdm(enumerate(data), desc='[{} batches]'.format(name)):
        optimizer.zero_grad()
        loss = tm(inputs_batch,
                targets_batch_out,
                input_length_batch,
                target_length_batch,
                pad_id, train=True)

        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().cpu().numpy()
        num_batches += 1
    return total_loss, num_batches

def train_truncation(args, tm, data):
    """
    Trains given truncation model and saves it to disk.

    The Adam optimizer is used. Loss is calculated by each model independently
    Args:
        args (dict): arguments from yaml config
        tm (models.Seq2Seq): model to train
        data (data.Seq2SeqDataset): contains the training and dev data
    """
    train_batches = data.get_train_dataloader(shuffle=True)
    dev_batches = data.get_dev_dataloader(shuffle=False)
    pad_id = data.target_vocab.word2idx['<pad>']
    min_dev_loss = float('inf')
    min_dev_loss_epoch = -1
    params_file = "{}/truncation_model.params".format(utils.get_trunc_model_path_of_args(args))

    # create optimizer
    optimizer = torch.optim.Adam(tm.parameters(), lr=args['truncation']['lr']) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)

    # calculate and report initial dev loss (with no training)
    epoch_train_loss, epoch_train_loss_count = (0, 1)
    with torch.no_grad():
        epoch_dev_loss, epoch_dev_loss_count = run_truncation_batches(dev_batches, tm, optimizer, "dev", pad_id, -1)
    tqdm.write("[Epoch: {}] Train loss: {}, Dev loss: {}".format(-1, epoch_train_loss, epoch_dev_loss/epoch_dev_loss_count))

    for epoch_idx in tqdm(range(MAX_EPOCHS), desc='[training truncation]'):
        # run training and development epochs
        epoch_train_loss, epoch_train_loss_count = run_truncation_batches(train_batches, tm, optimizer, "train", pad_id, epoch_idx)
        train_loss = epoch_train_loss / epoch_train_loss_count
        with torch.no_grad():
            epoch_dev_loss, epoch_dev_loss_count = run_truncation_batches(dev_batches, tm, optimizer, "dev", pad_id, epoch_idx)
            dev_loss = epoch_dev_loss / epoch_dev_loss_count

        # report training progress, update best model is needed
        tqdm.write(f"Train: {train_loss}\tDev: {dev_loss}")
        if dev_loss < min_dev_loss - MIN_DEV_LOSS_THRESHOLD: 
            min_dev_loss = dev_loss
            min_dev_loss_epoch = epoch_idx
            torch.save(tm.state_dict(), params_file)
            tqdm.write("Saving truncation model parameters")
        scheduler.step(epoch_dev_loss)

        if min_dev_loss_epoch < epoch_idx - EARLY_STOPPING_THRESHOLD:
            tqdm.write("Early stopping")
            tqdm.write("Min dev loss: {}".format(min_dev_loss))
            return
    tqdm.write("Min dev loss: {}".format(min_dev_loss))
