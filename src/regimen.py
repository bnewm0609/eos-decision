"""
Contains code for training sequence models.
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils

# -------- SEQ2SEQ --------
def run_seq2seq_batches(data, lm, optimizer, criterion, name):
    """
    Runs the model over an epoch of training/development data to calculate loss.

    Args:
        data (torch.utils.data.Dataloader): the train/dev data iterable
        lm (models.Seq2Seq): the model we're calculate the loss of
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Module): the loss function
        name (str): label for the batch: either "train" or "dev"
    Returns:
        tuple(float, int) with total loss (aggregated accross batches) and the
        total number of batches.
    """
    total_loss = 0
    num_batches = 0
    # we assume that not training means dev
    training = name == "train"
    if training:
        lm.train()
    else:
        lm.eval()
    for (
            batch_i,
            (inputs_batch,
            (targets_batch_in, targets_batch_out),
            (input_length_batch, target_length_batch), _)
            ) in tqdm(enumerate(data), desc='[{} batches]'.format(name)):
         
        # run batch through model and calculate loss
        optimizer.zero_grad()
        logits_batch = lm(inputs_batch, targets_batch_in, input_length_batch, target_length_batch)
        vocab_size = logits_batch.size(2)
        loss = criterion(logits_batch.reshape(-1, vocab_size), targets_batch_out.flatten())

        # only update parameters during training
        if training:
            loss.backward()
            optimizer.step()

        # track loss
        total_loss += loss.detach().cpu().numpy()
        num_batches += 1

    return total_loss, num_batches


def train(args, lm, data, batches_fn):
    """
    Trains given seq2seq model and saves it to disk.

    The Adam optimizer and cross-entropy loss is used.
    Args:
        args (dict): arguments from yaml config
        lm (models.Seq2Seq): model to train
        data (data.Seq2SeqDataset): contains the training and dev data
    """
    # Set up
    train_batches = data.get_train_dataloader(shuffle=True)
    dev_batches = data.get_dev_dataloader(shuffle=False)
    min_dev_ppl = float('inf')
    min_dev_loss_epoch = -1
    params_file = "{}/model.params".format(utils.get_lm_path_of_args(args))
    max_epochs = 100000

    # create optimizer
    optimizer = torch.optim.Adam(lm.parameters(), lr=args['lm']['lr']) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)

    # create loss function
    if args['data']['dataset_type'] == "dyckmk":
        pad_idx = data.vocab.word2idx['<pad>']
    else:
        pad_idx = data.target_vocab.word2idx['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # have to fill with params

    # calculate and report initial dev loss (PPl should be ~10)
    epoch_train_loss, epoch_train_loss_count = (0, 1)
    with torch.no_grad():
        epoch_dev_loss, epoch_dev_loss_count = batches_fn(dev_batches, lm, optimizer, criterion, "dev")
    train_ppl = np.exp(epoch_train_loss / epoch_train_loss_count)
    dev_ppl = np.exp(epoch_dev_loss / epoch_dev_loss_count)
    tqdm.write("[Epoch: {}] Train ppl: {}, Dev ppl: {}".format(-1, train_ppl, dev_ppl))

    for epoch_idx in tqdm(range(max_epochs), desc='[training seq2seq]'):
        # run training and development epochs
        epoch_train_loss, epoch_train_loss_count = batches_fn(train_batches, lm, optimizer, criterion, "train")
        with torch.no_grad():
            epoch_dev_loss, epoch_dev_loss_count = batches_fn(dev_batches, lm, optimizer, criterion, "dev")

        # report training progress, update best model is needed
        train_ppl = np.exp(epoch_train_loss / epoch_train_loss_count)
        dev_ppl = np.exp(epoch_dev_loss / epoch_dev_loss_count)
        tqdm.write("[Epoch: {}] Train ppl: {:.8f}, Dev ppl: {:.8f} ntrain:{}".format(epoch_idx, train_ppl, dev_ppl, epoch_train_loss_count))

        if dev_ppl < min_dev_ppl - 1e-4:
            min_dev_ppl = dev_ppl
            min_dev_loss_epoch = epoch_idx
            torch.save(lm.state_dict(), params_file)
            tqdm.write("Saving lm parameters")

        scheduler.step(epoch_dev_loss)

        # early stopping
        if min_dev_loss_epoch < epoch_idx - 5:
            tqdm.write("Early stopping")
            tqdm.write("Min dev ppl: {}".format(min_dev_ppl))
            return

    tqdm.write("Min dev ppl: {}".format(min_dev_ppl))


def train_seq2seq(args, lm, data):
    train(args, lm, data, run_seq2seq_batches)

# ------------------------------ Language Modeling ------------------------------


def run_lm_batches(data, lm, optimizer, criterion, name):
    """
    Runs the model over an epoch of training/development data to calculate loss.

    Args:
        data (torch.utils.data.Dataloader): the train/dev data iterable
        lm (models.Seq2Seq): the model we're calculate the loss of
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Module): the loss function
        name (str): label for the batch: either "train" or "dev"
    Returns:
        tuple(float, int) with total loss (aggregated accross batches) and the
        total number of batches.
    """
    total_loss = 0
    num_batches = 0
    # we assume that not training means dev
    training = name == "train"
    if training:
        lm.train()
    else:
        lm.eval()
    for batch_i, (inputs_batch, targets_batch, length_batch, _) in tqdm(
            enumerate(data), desc='[{} batches]'.format(name)):
         
        # run batch through model and calculate loss
        optimizer.zero_grad()
        logits_batch = lm(inputs_batch, length_batch)
        vocab_size = logits_batch.size(2)
        loss = criterion(logits_batch.reshape(-1, vocab_size), targets_batch.flatten())

        # only update parameters during training
        if training:
            loss.backward()
            optimizer.step()

        # track loss
        total_loss += loss.detach().cpu().numpy()
        num_batches += 1

    return total_loss, num_batches


def train_lm(args, lm, data):
    train(args, lm, data, run_lm_batches)


def run_validation(args, lm, dev_batches):
        epoch_dev_loss = 0
        epoch_dev_loss_count = 0
        hiddens_batch = None
        for batch_i, (observation_batch, label_batch, length_batch, _) in tqdm(enumerate(dev_batches), desc='[dev batches]'):
            lm.eval()
            logits_batch, hiddens_batch = lm(observation_batch, hiddens_batch)
            if args['data']['dataset_type'] == 'awd':
                hiddens_batch = utils.repackage_hidden(hiddens_batch)
            else:
                hiddens_batch = None
            total = 0
            ce_loss = 0
            for logits, target, length in zip(logits_batch, label_batch, length_batch):
                ce_loss += F.cross_entropy(logits[:length, :], target[:length], reduction='sum')
                total += length
            ce_loss /= total
            epoch_dev_loss += ce_loss.detach().cpu().numpy()
            epoch_dev_loss_count += 1

        return epoch_dev_loss, epoch_dev_loss_count
