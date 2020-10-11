"""
Contains the model classes for predicting where sequences end.
Right now is just probe for Dyck-k,m languages.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils


class TruncationModel(nn.Module):
    """
    Base class for truncation models that predict output lengths.
    """
    requires_lm = False

    def __init__(self, args):
        """
        Sets up the truncation model architecture

        Args:
            args (dict): arguments defined by config yaml
        """
        super().__init__()

    def forward(self, source_seq, target_seq, source_lens, target_lens, pad_idx, train=False):
        """
        Runs the forward pass of the model.
        
        Because not all of the truncation models are trained in the same way,
        the forward function also calculates the loss for each model if
        `train=True`.

        Args:
            source_seq (torch.Tensor(batch_size, max_src_seq_len)): batched pytorch
                tensor with sequence to encode.
            target_seq (torch.Tensor(batch_size, max_tgt_seq_len)): batched pytorch
                tensor with target or output sequence
            source_lens (torch.Tensor(batch_size)): lengths of source sequences
            target_lens (torch.Tensor(batch_size)): lengths of target sequences
            pad_idx (int): padding index in the vocabulary
            train (bool): flag indicating whether the model is being used at training or eval time
                TODO: depracate this and just call self.training


        Returns:
            if train is False, return the predicted lengths. Otherwise, return the loss from the batch
        """
        pass

    def load_pretrained_model(self, model_path=None):
        if model_path is None:
            model_path = "{}/truncation_model.params".format(utils.get_trunc_model_path_of_args(self.args))
        tqdm.write("Loading model from {}".format(model_path))
        self.load_state_dict(torch.load(model_path, map_location=self.args['device']))


class DyckKMProbePredictor(TruncationModel):

    requires_lm = True

    def __init__(self, args, lm):
        super().__init__(args)
        self.args = args
        self.lm = lm
        self.lm.eval()
        self.probe = torch.nn.Linear(args['lm']['hidden_dim'], 1)
        self.to(args["device"])


    def forward(self, source_seq, target_seq, source_lens, target_lens, pad_idx, train=False):
        with torch.no_grad():
            hidden_states, _ = self.lm.lm(self.lm.embed(source_seq))
        logits = self.probe(hidden_states)

        source_masks = -1000 * utils.generate_masks(source_lens).unsqueeze(2)

        if train:
            loss = F.binary_cross_entropy_with_logits((logits + source_masks).flatten(), target_seq.flatten())
            return loss
        else:
            predictions = (torch.sigmoid(logits) >= 0.5).int()
            return predictions + source_masks


class TruncationOracle(TruncationModel):

    """
    This is a dummy model that returns the true target length.

    It's useful for testing or getting the upper bound on our truncation
    model performances for SCAN.
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def forward(self, source_seq, target_seq, source_lens, target_lens, pad_idx, train=False):
        return target_lens

    def load_pretrained_model(self, model_path=None):
        pass


class TruncationIdentity(TruncationModel):

    """
    This is a dummy model that returns the true predicted sequence length.

    It's useful for testing.
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def forward(self, source_seq, target_seq, source_lens, target_lens, pad_idx, train=False):
        return len(target_seq)

    def load_pretrained_model(self, model_path=None):
        pass

