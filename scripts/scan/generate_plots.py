import sys; sys.path.append("src/")

from argparse import ArgumentParser
from collections import namedtuple
from itertools import chain
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import subprocess
import torch
from torch.nn.functional import softmax

from data import DatasetSCAN
from models import Seq2SeqLSTM
from utils import load_config_from_path, get_results_dir_of_args

matplotlib.use('agg')
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['figure.titlesize'] = 'medium'

SCAN_HS = namedtuple("SCAN_hidden_states", ["train", "dev", "test"])
COLOR_MAP = 'cool'

FIGURE_PATH = "results/plots/scan/{}_plot.png"
NO_EOS_CONFIG = "configs/scan/plots/scan_-EOS.yaml"
EOS_CONFIG = "configs/scan/plots/scan_+EOS.yaml"

NO_EOS_DIR = get_results_dir_of_args(load_config_from_path(NO_EOS_CONFIG)) # "results/ls-22_eos-F/scan-scanls22-bs16_dec_cellstandard_dec_num_layers1_dropout0.5_embed_dim200_enc_cellstandard_enc_num_layers2_hidden_dim200_seq2seq_lr0.001_seed500_tfr0.5_eosF-model_typeoracle_truncation_pathmodels/ls-22_eos-F"
EOS_DIR = get_results_dir_of_args(load_config_from_path(EOS_CONFIG)) # "results/ls-22_eos-T/scan-scanls22-bs16_dec_cellstandard_dec_num_layers1_dropout0.5_embed_dim200_enc_cellstandard_enc_num_layers2_hidden_dim200_seq2seq_lr0.001_seed500_tfr0.5_eosT-model_typeoracle_truncation_pathmodels/ls-22_eos-T"


def get_sequence_cutoffs(sequences):
    cum_lens = np.cumsum([len(seq) for seq in sequences]).tolist()
    return [slice(start, end) for start, end in zip([0] + cum_lens[:-1], cum_lens)]

def load_observations(data_loader):
    all_observations = []
    for _, _, _, observations in data_loader:
        all_observations.extend([obs.target_tokens[1:] for obs in observations])
    return all_observations

def slices_to_ints(slices_list):
    arr_len = sum([s.stop - s.start for s in slices_list])
    arr = np.empty(arr_len).astype(int)
    arr_ind = 0
    for s in slices_list:
        s_len = s.stop - s.start
        arr[arr_ind: arr_ind + s_len] = np.arange(s.start, s.stop)
        arr_ind += s_len
    return arr

# def shuffle(hs, data, slices, seed=1):
#     np.random.seed(seed); shuffle_train = np.random.permutation(len(data.train))
#     np.random.seed(seed); shuffle_dev = np.random.permutation(len(data.dev))
#     np.random.seed(seed); shuffle_test = np.random.permutation(len(data.test))

#     shuffle_slices = SCAN_HS(
#         np.array(slices.train)[shuffle_train],
#         np.array(slices.dev)[shuffle_dev],
#         np.array(slices.test)[shuffle_test],
#     )

#     shuffle_hs = SCAN_HS(
#         hs.train[slices_to_ints(shuffle_slices.train)],
#         hs.dev[slices_to_ints(shuffle_slices.dev)],
#         hs.test[slices_to_ints(shuffle_slices.test)],
#     )

#     shuffle_data = SCAN_HS(
#         np.array(data.train)[shuffle_train].tolist(),
#         np.array(data.dev)[shuffle_dev].tolist(),
#         np.array(data.test)[shuffle_test].tolist(),
#     )

#     # now re-do the slices to match up with the correct indices
#     shuffle_slices = SCAN_HS(
#         get_sequence_cutoffs(shuffle_data.train),
#         get_sequence_cutoffs(shuffle_data.dev),
#         get_sequence_cutoffs(shuffle_data.test),
#     )

#     return shuffle_hs, shuffle_data, shuffle_slices


def get_token_idxs(data):
    token_idxs = []
    for seq in data:
        token_idxs.extend(np.arange(len(seq)))
    return np.array(token_idxs)


def get_data(config, results_dir):
    config = load_config_from_path(
            config
    )
    data_raw = DatasetSCAN(config)
    data = SCAN_HS(
        load_observations(data_raw.get_train_dataloader(shuffle=False)),
        load_observations(data_raw.get_dev_dataloader(shuffle=False)),
        load_observations(data_raw.get_test_dataloader(shuffle=False)),
    )

    hidden_states = SCAN_HS(np.load(f"{results_dir}/cache.train.npy"),
               np.load(f"{results_dir}/cache.dev.npy"),
               np.load(f"{results_dir}/cache.test.npy"))

    # import ipdb; ipdb.set_trace()
    sent_idx_slices = SCAN_HS(
        get_sequence_cutoffs(data.train),
        get_sequence_cutoffs(data.dev),
        get_sequence_cutoffs(data.test),
    )

    token_idxs = SCAN_HS(None, get_token_idxs(data.dev), get_token_idxs(data.test))
    return config, data_raw, data, hidden_states, sent_idx_slices, token_idxs


def dimension_reduce(scan_hs, num_samples=10000, method=PCA, **kwargs):
    if num_samples == "all":
        num_samples = max(scan_hs.train.shape[0], scan_hs.test.shape[0])
    dim_red = method(n_components=2, **kwargs)
    train_red = dim_red.fit_transform(scan_hs.train[:num_samples])
    try:
        dev_red = dim_red.transform(scan_hs.dev[:num_samples])
        test_red = dim_red.transform(scan_hs.test[:num_samples])
    except AttributeError:
        dev_red = np.zeros((1, 2))
        test_red = np.zeros((1, 2))
    return SCAN_HS(train_red, dev_red, test_red), dim_red


def new_fig():
    fig = plt.figure(figsize=(5,6), dpi=200)
    ax = fig.add_subplot(111)
    return fig, ax


def plot_hs_length_grad(scan_hs, token_idxs, titles, alphas, num_samples=10000, out_filename=""):
    assert len(token_idxs.dev) == len(scan_hs.dev) and len(token_idxs.test) == len(scan_hs.test)
    cm = matplotlib.cm.get_cmap(COLOR_MAP)
    
    devfig, devax = new_fig()
    devax.scatter(*scan_hs.train[:num_samples].T, color="grey", label="Train", alpha=alphas['train'])
    dev_scatter = devax.scatter(*scan_hs.dev[:num_samples].T, c=token_idxs.dev[:num_samples], cmap=cm, alpha=alphas['dev'])
    # add colorbar
    colorbar = devfig.colorbar(dev_scatter, ax=devax, fraction=0.046, pad=0.04)
    colorbar.solids.set(alpha=1)
    devax.set_title(titles[0], fontsize=16)
    legend = devax.legend(loc=(0, 0))
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    devfig.savefig(FIGURE_PATH.format(f"{out_filename}_dev"), bbox_inches='tight')
    
    testfig, testax = new_fig()
    testax.scatter(*scan_hs.train[:num_samples].T, color="grey", label="Train", alpha=alphas['train'])
    test_scatter = testax.scatter(*scan_hs.test[:num_samples].T, c=token_idxs.test[:num_samples], cmap=cm, alpha=alphas['test'])
    colorbar = testfig.colorbar(test_scatter, ax=testax, fraction=0.046, pad=0.04)
    colorbar.solids.set(alpha=1)
    testax.set_title(titles[1], fontsize=16)
    legend = testax.legend(loc=(0, 0))
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    testfig.savefig(FIGURE_PATH.format(f"{out_filename}_test"), bbox_inches='tight')
    

def plot_eos_vs_not_hs(scan_hs, scan_probs, title, num_samples=10000, train_alpha=0.05, eos_alpha=0.1, no_eos_alpha=0.1, eos_idx=0, out_filename=""):
    if num_samples == "all":
        num_samples = np.max([len(scan_hs.train), len(scan_hs.dev), len(scan_hs.test)])

    # we only care about TEST probabilities
    eos_preds = np.argmax(scan_probs.test[:num_samples], axis=1) == eos_idx

    plt.scatter(*scan_hs.train[:num_samples].T, color="grey", label="Train", alpha=train_alpha)
    plt.scatter(*scan_hs.test[:num_samples][~eos_preds].T, color="lightblue", label="Not <EOS>", alpha=no_eos_alpha)
    plt.scatter(*scan_hs.test[:num_samples][eos_preds].T, color="red", label="<EOS>", alpha=eos_alpha)

    plt.title(title, fontsize=16)
    legend = plt.legend(loc=(1.04, 0), fontsize=14)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.savefig(FIGURE_PATH.format(out_filename), bbox_inches="tight")
    plt.clf()


def _plot_token_hs(train, test, flat_tokens, title, num_samples, train_alpha, test_alpha, out_filename=""):
    vocab = set(flat_tokens)
    
    # ordering things for the labels
    is_eos = '<eos>' in vocab
    if is_eos:
        vocab.remove('<eos>')
    vocab = sorted(list(vocab))
    if is_eos:
        vocab.append('<eos>')
    
    plt.scatter(*train[:num_samples].T, color="grey", label="Train", alpha=train_alpha)
    for token in vocab:
        lab = (token[2:] if token.startswith('I_') else token).upper()
        plt.scatter(*test[flat_tokens == token].T, label=lab, alpha=test_alpha)
    plt.title(title, fontsize=16)
    legend = plt.legend(loc=(1.04, 0), fontsize=14)
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    plt.savefig(FIGURE_PATH.format(out_filename), bbox_inches="tight")
    plt.clf()

    
def plot_token_hs(scan_hs, data, title, num_samples=10000, alpha=(0.01, 0.01), out_filename=""):
    if num_samples == "all":
        num_samples = np.max([len(scan_hs.train), len(scan_hs.dev), len(scan_hs.test)])
    flat_dev_data = np.array(list(chain(*data.dev)))[:num_samples]
    _plot_token_hs(scan_hs.train, scan_hs.dev[:num_samples], flat_dev_data, f"SCAN In Domain {title}", num_samples, alpha[0], alpha[1], out_filename=f"{out_filename}_token_id_dev")
    
    flat_test_data = np.array(list(chain(*data.test)))[:num_samples]
    _plot_token_hs(scan_hs.train, scan_hs.test[:num_samples], flat_test_data, f"SCAN Out of Domain {title}", num_samples, alpha[0], alpha[1], out_filename=f"{out_filename}_token_id_dev")


def _plot_sequence_hs(train, sequence, title, num_samples, alpha, out_filename=""):
    plt.scatter(*train[:num_samples].T, color="grey", alpha=alpha)
    plt.scatter(*sequence.T, color="red", alpha=1)
    plt.title(title, fontsize=16)
    for i, point in enumerate(sequence):
        plt.annotate(f"{i+1}", (point[0], point[1]))
    plt.savefig(FIGURE_PATH.format(out_filename), bbox_inches="tight")
    plt.clf()


def plot_sequence_hs(scan_hs, seq_idxs, slices, data, title, num_samples=10000, alpha=0.01, out_filename=""):
    dev_seq_idx, test_seq_idx = seq_idxs
    # import ipdb; ipdb.set_trace()

    # id
    if 0 <= dev_seq_idx < len(data.dev):
        dev_seq_hs = scan_hs.dev[slices.dev[dev_seq_idx]]
        _plot_sequence_hs(scan_hs.train, dev_seq_hs, title, num_samples, alpha, out_filename=out_filename)
    
    # odd
    if 0 <= test_seq_idx < len(scan_hs.test):
        test_seq_hs = scan_hs.test[slices.test[test_seq_idx]]
        _plot_sequence_hs(scan_hs.train, test_seq_hs, title, num_samples, alpha, out_filename=out_filename)


def plot_sequences_paper(sentence_id, length, hs, slices, data, eos):
    pca_hs, big_pca = dimension_reduce(hs, num_samples="all")
    
    if length > 22:
        plot_sequence_hs(pca_hs, (-1, sentence_id), slices, data, f"SCAN ({eos}) Sequence Length {len(data.test[sentence_id])}", out_filename=f"{eos}_{length}")
    else:
        plot_sequence_hs(pca_hs, (sentence_id, -1), slices, data, f"SCAN ({eos}) Sequence Length {len(data.dev[sentence_id])}", out_filename=f"{eos}_{length}")


@torch.no_grad()
def get_probs(model, hs, batch_size=100):
    probs = []
    for batch_i in range(0, len(hs), batch_size):
        probs.extend(
            softmax(model.target_vocab_proj(
                torch.from_numpy(hs[batch_i : batch_i + batch_size])
                ), dim=1).cpu().numpy()
        )
    return np.stack(probs)


def generate_plots():
    noeos_config, noeos_data_raw, noeos_data, noeos_hs, noeos_sent_idx_slices, noeos_token_idxs = get_data(NO_EOS_CONFIG, NO_EOS_DIR)
    eos_config, eos_data_raw, eos_data, eos_hs, eos_sent_idx_slices, eos_token_idxs = get_data(EOS_CONFIG, EOS_DIR)

    # +EOS plot by token id
    eos_pca_hs, _ = dimension_reduce(eos_hs)
    plot_token_hs(eos_pca_hs, eos_data, "(+EOS)", alpha = (0.04, 0.1), out_filename="+EOS")

    # -EOS plot by token id
    noeos_pca_hs, _ = dimension_reduce(noeos_hs)
    plot_token_hs(noeos_pca_hs, noeos_data, "(-EOS)", alpha = (0.04, 0.1), out_filename="-EOS")
    print("[generated hidden states plots (colored by token identity)]")


    # +EOS plot by EOS token or not
    eos_model = Seq2SeqLSTM(eos_config)
    eos_model.load_pretrained_model()
    eos_model_probs = SCAN_HS(None, None, get_probs(eos_model, eos_hs.test))
    eos_pca_hs, _ = dimension_reduce(eos_hs, num_samples=10000)
    plot_eos_vs_not_hs(eos_pca_hs, eos_model_probs, "SCAN Out of Domain (+EOS)", no_eos_alpha=0.1, out_filename="+eos_ood_eos_or_not")

    # -EOS plot by EOS token or not
    noeos_model = Seq2SeqLSTM(noeos_config)
    noeos_model.load_pretrained_model()
    noeos_model_probs = SCAN_HS(None, None, get_probs(noeos_model, noeos_hs.test))
    noeos_pca_hs, _ = dimension_reduce(noeos_hs, num_samples=10000)
    plot_eos_vs_not_hs(noeos_pca_hs, noeos_model_probs, "SCAN Out of Domain (-EOS)", no_eos_alpha=0.1, eos_alpha=0.1, out_filename="-eos_ood_eos_or_not")
    print("[generated hidden states plots (colored by EOS predicted or not)]")

    # +EOS plot by length
    eos_pca_hs, _ = dimension_reduce(eos_hs, "all")
    lengrad_alphas = {'train': 0.01, 'dev': 0.1, 'test': 0.05}
    eos_titles = ("SCAN In Domain (+EOS)", "SCAN Out of Domain (+EOS)")
    plot_hs_length_grad(eos_pca_hs, eos_token_idxs, eos_titles, lengrad_alphas, out_filename="+EOS_length")

    # -EOS plot by length
    noeos_pca_hs, _ = dimension_reduce(noeos_hs, "all")
    lengrad_alphas = {'train': 0.01, 'dev': 0.1, 'test': 0.05}
    noeos_titles = ("SCAN In Domain (-EOS)", "SCAN Out of Domain (-EOS)")
    plot_hs_length_grad(noeos_pca_hs, noeos_token_idxs, noeos_titles, lengrad_alphas, out_filename="-EOS_length")
    print("[generated hidden states plots (colored by token index)]")

    # +EOS example paths
    lengths = [4, 16, 24, 33, 48]
    sentence_ids = [625, 28, 3602, 400, 972]
    for sentence_id, length in zip(sentence_ids, lengths):
        plot_sequences_paper(sentence_id, length, eos_hs, eos_sent_idx_slices, eos_data, eos="+EOS")
        plot_sequences_paper(sentence_id, length, noeos_hs, noeos_sent_idx_slices, noeos_data, eos="-EOS")
    print("[generated example paths through hidden state space]")


def main():
    os.makedirs("results/plots/scan/", exist_ok=True)
    argp = ArgumentParser()
    argp.add_argument("--save_hidden_states", action="store_true")
    args = argp.parse_args()
    if args.save_hidden_states:
        subprocess.run(f"python src/run.py {EOS_CONFIG}", shell=True)
        subprocess.run(f"python src/run.py {NO_EOS_CONFIG}", shell=True)
    generate_plots()

if __name__ == "__main__":
    main()
