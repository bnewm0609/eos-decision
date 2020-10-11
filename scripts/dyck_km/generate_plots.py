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

from data import Dataset
from utils import load_config_from_path

matplotlib.use('agg')
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['figure.titlesize'] = 'medium'

SCAN_HS = namedtuple("SCAN_hidden_states", ["train", "dev", "test"])
COLOR_MAP = 'cool'

FIGURE_PATH = "results/plots/dyck24/{}_plot_{}.png"
EOS_CONFIG = "configs/dyck_km/plots/dyck24_+EOS.yaml"
EOS_DIR = "results/dyck42-eos/dyckmk-dyck42-bs16_cellstandard_dropout0.0_embed_dim20_hidden_dim20_rnnlm_lr0.01_num_layers1_seed501_eosT-"
NO_EOS_CONFIG = "configs/dyck_km/plots/dyck24_-EOS.yaml"
NO_EOS_DIR = "results/dyck42-no_eos/dyckmk-dyck42-bs16_cellstandard_dropout0.0_embed_dim20_hidden_dim20_rnnlm_lr0.01_num_layers1_seed501_eosF-"

def get_sequence_cutoffs(sequences):
    cum_lens = np.cumsum([len(seq) for seq in sequences]).tolist()
    return [slice(start, end) for start, end in zip([0] + cum_lens[:-1], cum_lens)]


def load_observations(data_loader):
    all_observations = []
    for _, _, _, observations in data_loader:
        all_observations.extend([obs.tokens[:-1] for obs in observations])
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


def get_token_idxs(data):
    token_idxs = []
    for seq in data:
        token_idxs.extend(np.arange(len(seq)))
    return np.array(token_idxs)


def get_data(config, results_dir):
    config = load_config_from_path(
            config
    )
    data_raw = Dataset(config)
    data = SCAN_HS(
        load_observations(data_raw.get_train_dataloader(shuffle=False)),
        load_observations(data_raw.get_dev_dataloader(shuffle=False)),
        load_observations(data_raw.get_test_dataloader(shuffle=False)),
    )

    hidden_states = SCAN_HS(np.load(f"{results_dir}/cache.train.npy"),
               np.load(f"{results_dir}/cache.dev.npy"),
               np.load(f"{results_dir}/cache.test.npy"))

    sent_idx_slices = SCAN_HS(
        get_sequence_cutoffs(hidden_states.train),
        get_sequence_cutoffs(hidden_states.dev),
        get_sequence_cutoffs(hidden_states.test),
    )

    token_idxs = SCAN_HS(None, get_token_idxs(data.dev), get_token_idxs(data.test))
    return data_raw, data, hidden_states, sent_idx_slices, token_idxs


def plot_hs(scan_hs, title_id, title_ood, num_samples=10000, alpha=0.01):
    if isinstance(alpha, tuple):
        tr_a, d_a, te_a = alpha
    else:
        tr_a = d_a = te_a = alpha

    plt.scatter(*scan_hs.train[:num_samples].T, color="grey", alpha=tr_a)
    plt.scatter(*scan_hs.dev[:num_samples].T, color="red", alpha=d_a)
    plt.title(f"{title} (in domain)")
    plt.show()
    plt.scatter(*scan_hs.train[:num_samples].T, color="grey", alpha=tr_a)
    plt.scatter(*scan_hs.test[:num_samples].T, color="red", alpha=te_a)
    plt.title(f"{title} (out of domain)")
    plt.show()


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
    devfig.savefig(FIGURE_PATH.format(out_filename, "dev"), bbox_inches='tight')
    
    testfig, testax = new_fig()
    testax.scatter(*scan_hs.train[:num_samples].T, color="grey", label="Train", alpha=alphas['train'])
    test_scatter = testax.scatter(*scan_hs.test[:num_samples].T, c=token_idxs.test[:num_samples], cmap=cm, alpha=alphas['test'])
    colorbar = testfig.colorbar(test_scatter, ax=testax, fraction=0.046, pad=0.04)
    colorbar.solids.set(alpha=1)
    testax.set_title(titles[1], fontsize=16)
    legend = testax.legend(loc=(0, 0))
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    testfig.savefig(FIGURE_PATH.format(out_filename, "test"), bbox_inches='tight')
    

def get_stack_states(sequences):
    state_labels = []
    for sentence in sequences:
        state_vec = []
        subseq_vec = []
        for char in sentence[:-1]:
            # subseq_vec.append(char)
            if "END" in char:
                break
            if "(" in char:
                state_vec.append(char.strip('('))
            if ")" in char:
                state_vec = state_vec[:-1]
            state_labels.append("".join(state_vec))
    return np.array(state_labels)


def get_stack_state_filter(raw_data, stack_state):
    train_data = [ex.tokens for ex in raw_data.train_data]
    dev_data = [ex.tokens for ex in raw_data.dev_data]
    test_data = [ex.tokens for ex in raw_data.test_data]
    return SCAN_HS(np.array(get_stack_states(train_data)) == stack_state,
                   np.array(get_stack_states(dev_data)) == stack_state,
                   np.array(get_stack_states(test_data)) == stack_state
                  )


def filter_state_and_len(hs, raw_data, token_idxs, state, max_len=120, num_samples = 10000):
    pca_hs, _ = dimension_reduce(hs, num_samples=num_samples)
    stack_state_filter = get_stack_state_filter(raw_data, state)
    full_filter = SCAN_HS(
        stack_state_filter.train,
        stack_state_filter.dev & (token_idxs.dev <=  max_len),
        stack_state_filter.test & (token_idxs.test <= max_len),
    )

    pca_hs = SCAN_HS(
        pca_hs.train[full_filter.train[:num_samples]],
        pca_hs.dev[full_filter.dev[:num_samples]],
        pca_hs.test[full_filter.test[:num_samples]],
    )
    filtered_token_idxs = SCAN_HS(
        None,
        token_idxs.dev[:num_samples][full_filter.dev[:num_samples]],
        token_idxs.test[:num_samples][full_filter.test[:num_samples]],
    )
    return pca_hs, filtered_token_idxs


def plot_hs_token_id(dyck_hs, dyck_states, titles, alphas, num_samples=10000, mult=1, outfile_name=""):
    all_states = set(dyck_states.dev)
    all_states.remove('')
    all_states.add('[EMPTY]')
    all_states = sorted(list(all_states), key=lambda x: x[::-1])

    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(all_states)))
    # order colors by the REVERSE sorted order of the state
    color_map = dict(zip(all_states, colors))
    color_map['[EMPTY]'] = 'brown'

    # DEV
    devfig, devax = new_fig()
    for state in all_states:
        if state == "b": # halfway
            l = devax.scatter([0],[0],color="w", label=" ")
        subset_hs = dyck_hs.train[dyck_states.train == (state if state != "[EMPTY]" else "")][:num_samples]
        dev_scatter = devax.scatter(*(mult * subset_hs.T),
                                    label=state,
                                    alpha=alphas['train'],
                                    color=color_map[state])

    devax.set_title(titles[0], fontsize=16)
    legend = devax.legend(loc=(1.04, 0), ncol=2, title="Stack State")
    plt.setp(legend.get_title(),fontsize='large')
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    devfig.savefig(FIGURE_PATH.format(outfile_name, "dev"), bbox_inches='tight')


def generate_plots():
    noeos_data_raw, noeos_data, noeos_hs, noeos_sent_idx_slices, noeos_token_idxs = get_data(NO_EOS_CONFIG, NO_EOS_DIR)
    eos_data_raw, eos_data, eos_hs, eos_sent_idx_slices, eos_token_idxs = get_data(EOS_CONFIG, EOS_DIR)

    # generate +EOS plots (all hidden states)
    eos_pca_hs, _ = dimension_reduce(eos_hs, "all")
    lengrad_alphas = {'train': 0.01, 'dev': 0.1, 'test': 0.05}
    eos_titles = ("Dyck-2,4 In Domain (+EOS)", "Dyck-2,4 Out of Domain (+EOS)")
    plot_hs_length_grad(eos_pca_hs, eos_token_idxs, eos_titles, lengrad_alphas, out_filename="eos_all")

    # generate -EOS plots (all hidden states)
    noeos_pca_hs, _ = dimension_reduce(noeos_hs, num_samples="all")
    lengrad_alphas = {'train': 0.01, 'dev': 0.1, 'test': 0.05}
    noeos_titles = ("Dyck-2,4 In Domain (-EOS)", "Dyck-2,4 Out of Domain (-EOS)")
    plot_hs_length_grad(noeos_pca_hs, noeos_token_idxs, noeos_titles, lengrad_alphas, out_filename="no_eos_all")
    print("[generated hidden states plots]")


    # generate +EOS plots (B hidden states)
    eos_pca_SSb, eos_token_idxs_SSb = filter_state_and_len(
            eos_hs, eos_data_raw, eos_token_idxs, "b", num_samples=1000000
            )
    lengrad_b_alphas = {"train": 0.05, "dev": 0.5, "test": 0.1}
    eos_titles_SSb = ("Dyck-2,4 (+EOS)", "Dyck-2,4 (+EOS)")
    plot_hs_length_grad(eos_pca_SSb, eos_token_idxs_SSb, eos_titles_SSb, lengrad_b_alphas, out_filename="eos_b")


    # generate -EOS plots (B hidden states)
    noeos_pca_SSb, noeos_token_idxs_SSb = filter_state_and_len(
            noeos_hs, noeos_data_raw, noeos_token_idxs, "b", num_samples=100000
            )
    lengrad_b_alphas = {"train": 0.05, "dev": 0.5, "test": 0.2}
    noeos_titles_SSb = ("Dyck-2,4 (-EOS)", "Dyck-2,4 (-EOS)")
    plot_hs_length_grad(noeos_pca_SSb, noeos_token_idxs_SSb, noeos_titles_SSb, lengrad_b_alphas, out_filename="no_eos_b")
    print("[generated hidden states plots (single stack state)]")


    # generate +EOS plots (color stacks)
    eos_stack_states = SCAN_HS(
        get_stack_states([x.tokens for x in eos_data_raw.train_data]),
        get_stack_states([x.tokens for x in eos_data_raw.dev_data]),
        get_stack_states([x.tokens for x in eos_data_raw.test_data]),
    )
    eos_pca_hs, _ = dimension_reduce(eos_hs, num_samples="all")
    eos_titles_stack_states = ("Dyck-2,4 (+EOS)", "Dyck-2,4 (+EOS)")
    eos_alpha = {"train": 0.05, "dev": 0.05, "test": 0.01}
    plot_hs_token_id(eos_pca_hs, eos_stack_states, eos_titles_stack_states, eos_alpha, mult=-1, outfile_name="eos_stack_state")


    # generate -EOS plots (color stacks)
    noeos_stack_states = SCAN_HS(
        get_stack_states([x.tokens for x in noeos_data_raw.train_data]),
        get_stack_states([x.tokens for x in noeos_data_raw.dev_data]),
        get_stack_states([x.tokens for x in noeos_data_raw.test_data]),
    )
    noeos_pca_hs, _ = dimension_reduce(noeos_hs, num_samples="all")
    noeos_titles_stack_states = ("Dyck-2,4 (-EOS)", "Dyck-2,4 (-EOS)")
    noeos_alpha = {"train": 0.05, "dev": 0.05, "test": 0.01}
    plot_hs_token_id(noeos_pca_hs, noeos_stack_states, noeos_titles_stack_states, noeos_alpha, outfile_name="no_eos_stack_state")
    print("[generated hidden states plots (colored by stack state)]")



def main():
    os.makedirs("results/plots/dyck24/", exist_ok=True)
    argp = ArgumentParser()
    argp.add_argument("--save_hidden_states", action="store_true")
    args = argp.parse_args()
    if args.save_hiddens:
        subprocess.run(f"python src/run.py {EOS_CONFIG}", shell=True)
        subprocess.run(f"python src/run.py {NO_EOS_CONFIG}", shell=True)
    generate_plots()

if __name__ == "__main__":
    main()
