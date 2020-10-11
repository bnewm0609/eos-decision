"""
Contains classes for reporting results from a single experiment.
"""
from collections import defaultdict
import json
import numpy as np
import os
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sacrebleu 

from utils import get_results_dir_of_args, DETOKENIZER_FILE



class Reporter:
    """Base class for all reporters"""
    def __init__(self, args, name, data, *extra_args, **kwargs):
        self.args = args
        self.split_name = name
        self.data = data

    def update(self, batch_data):
        """
        Updates internal state and statistics with batch_data
        
        Args:
            batch_data (tuple): contains predicted outputs and targets
                in different forms depending on the type of reporter.

                - Standard reporters:
                (logits, targets, target_lengths, observations, predicted_lengths)
                - Sampling reporters:
                (predicted_sequences, targets, target_lengths, observations, predicted_lengths)
        """
        pass

    def report(self):
        """Summarizes internal state and outputs to the console and/or disk"""
        pass


class BleuReporter(Reporter):
    """Calculates the corpus BLEU score between the target and decoded sequences."""
    def __init__(self, args, name, data, **kwargs):
        super().__init__(args, name, data)
        self.target_lang_code = self.args['data']['target_lang']
        self.predictions = []
        self.references = []
        self.use_eos = self.args['lm']['use_eos']
        self.des = self.args['lm'].get('disable_eos_sampling', False)

    def update(self, batch_data):
        outputs_batch, targets_batch, target_lengths_batch, observations_batch, predicted_lengths_batch  = batch_data

        for output, target, target_length, pred_length in zip(outputs_batch, targets_batch, target_lengths_batch, predicted_lengths_batch):
            if target_length > 100:
                continue
            if self.use_eos:
                target_length = target_length.item() - 1  # Don't use EOS in BLEU score
                if not self.des:
                    pred_length = len(output) - 1 # -1 because of the extra <eos>

            self.predictions.append(" ".join([self.data.target_vocab.idx2word[i.item()] for i in output[:pred_length]]))
            self.references.append(" ".join([self.data.target_vocab.idx2word[i.item()] for i in target[:target_length]]))

    def detokenize(self, sentences):
        """
        Runs the moses perl detokenizer script on the tokenized target and output.

        Args: 
            sentences is List[str]
        Returns:
            list of detokenized output
        """
        detok_proc = subprocess.Popen(
                ["perl", DETOKENIZER_FILE, f"-l {self.target_lang_code}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True)
        outputs = detok_proc.communicate(input="\n".join(sentences))[0]
        return outputs.strip().split("\n")[2:]

    def report(self):
        """
        Runs the detokenizer and then sacrebleu to calculate the score.
        """
        predictions = self.detokenize(self.predictions)
        references = self.detokenize(self.references)
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        tqdm.write("Bleu score: {}".format(bleu.score))
        outfile_name = os.path.join(get_results_dir_of_args(self.args), "bleu.{}".format(self.split_name))
        with open(outfile_name, "w") as f:
            try:
                f.write("{}\n{}\n".format(bleu.score, str(bleu)))
            except ZeroDivisionError:
                f.write("{}\nNA\n".format(bleu.score))


class CacheReporter(Reporter):
    """
    Saves hidden states stored in cache to disk along with the key
    vectors used when computing nearest neighbors.
    """
    def __init__(self, args, name, data, lm):
        super().__init__(args, name, data)
        self.lm = lm
        lm.record_hidden_states()


    def report(self):
        cache = self.lm.dump_cache(clear=False)
        outfile_name = "{}/cache.{}.npy".format(get_results_dir_of_args(self.args), self.split_name)
        np.save(outfile_name, cache)


class CloseBracketReporter(Reporter):
    """
    Reports the percentage of closing brackets that have > 80% probability mass
    on the correct state.
    """

    def __init__(self, args, name, data):
        super().__init__(args, name, data)
        self.total_correct = defaultdict(int)
        self.total = defaultdict(int)

        closing_bracket_ids = []
        for token, token_id in data.vocab.word2idx.items():
            if ")" in token:
                closing_bracket_ids.append(token_id)
        self.closing_bracket_ids = torch.tensor(closing_bracket_ids).to(args['device'])

    def update(self, batch_data):
        logits_batch, targets_batch, lengths_batch, observations_batch = batch_data
        logits_batch = F.softmax(logits_batch, dim=2) # TODO: confirm
        
        for dist, observations in zip(logits_batch, observations_batch):
            state = [(0, observations.tokens[0][1])]
            for token_i, token in enumerate(observations.tokens[1:]):

                if state:
                    # we can close a bracket
                    total_close_prob_mass = torch.sum(dist[token_i, self.closing_bracket_ids])
                    prev_idx, prev_token = state[-1]
                    correct_close_token = f"{prev_token})"
                    correct_close_prob_mass = dist[token_i, self.data.vocab.word2idx[correct_close_token]]
                    self.total_correct[token_i - prev_idx] += int(correct_close_prob_mass >= 0.8 * total_close_prob_mass)
                    self.total[token_i - prev_idx] += 1

                # track state:
                if ")" in token:
                    state = state[:-1]
                else:
                    state.append((token_i + 1, token[1]))

    def report(self):
        outfile_name = "{}/close_bracket_acc.{}".format(get_results_dir_of_args(self.args), self.split_name)
        output = []
        all_results = []
        for key in sorted(self.total):
            # if self.total[key] >= 5:
            all_results.append(self.total_correct[key]/self.total[key])
            output.append(f"{key},{self.total_correct[key]/self.total[key]}")
        result_min = np.min(all_results)
        result_median = np.median(all_results)
        result_1pt = np.percentile(all_results, [.25])
        summary_str = f"{result_median} {result_1pt} {result_min}"

        tqdm.write("Weighted Closing Bracket Accuracy ({}): {}".format(self.split_name, summary_str))
        output_str = "{}\n{}\n".format("\n".join(output), summary_str)
        with open(outfile_name, "w") as f:
            f.write(output_str)

class DyckEndingReporter(Reporter):
    """
    Reports the percentage of closing brackets that have > 80% probability mass
    on the correct state.
    """

    def __init__(self, args, name, data):
        super().__init__(args, name, data)
        self.total_correct = 0
        self.total = 0

        closing_bracket_ids = []
        for token, token_id in data.vocab.word2idx.items():
            if ")" in token:
                closing_bracket_ids.append(token_id)
        self.closing_bracket_ids = torch.tensor(closing_bracket_ids).to(args['device'])

    def update(self, batch_data):
        logits_batch, targets_batch, truncation_preds_batch, observations_batch = batch_data
        logits_batch = F.softmax(logits_batch, dim=2)
        
        for truncation_preds, targets, observations in zip(truncation_preds_batch, targets_batch, observations_batch):
            seq_len = len(observations.source_tokens) - 1
            self.total_correct += torch.sum(torch.isclose(truncation_preds[:seq_len].flatten(), targets[:seq_len])).cpu().item()
            self.total += seq_len

    def report(self):
        outfile_name = "{}/dyck_ending_acc.{}".format(get_results_dir_of_args(self.args), self.split_name)
        summary_str = f"{self.total_correct / self.total}"
        tqdm.write("Ending Accuracy ({}): {}".format(self.split_name, summary_str))
        output_str = "{}\n".format(summary_str)
        with open(outfile_name, "w") as f:
            f.write(output_str)


class ExactMatchReporter(Reporter):
    """
    Reports the fraction of outputs that exactly match the targets. (Used for SCAN)
    """
    def __init__(self, args, name, data, **kwargs):
        self.args = args
        self.name = name
        self.total_correct = 0
        self.total = 0
    
    def update(self, batch_data):
        outputs_batch, targets_batch, target_lengths_batch, observations_batch, _ = batch_data
        for output, target, target_length in zip(outputs_batch, targets_batch, target_lengths_batch):
            if target_length == len(output) and torch.all(torch.eq(output, target[:target_length])):
                self.total_correct += 1
            self.total += 1

    def report(self):
        outfile_name = "{}/exact_match_acc.{}".format(get_results_dir_of_args(self.args), self.name)
        accuracy = self.total_correct / self.total
        tqdm.write("Exact match accuracy ({}): {} (total: {})".format(self.name, accuracy, self.total))
        with open(outfile_name, "w") as f:
            f.write("{}\n".format(accuracy))


class OracleBleuReporter(BleuReporter):
    """
    Try all possible endings and choose the one with the best sentence-level bleu score.

    NOTE: We assume that the outputs are already detokenized.
    """
    def __init__(self, args, name, data, **kwargs):
        super().__init__(args, name, data)
        self.target_lang_code = self.args['data']['target_lang']
        self.predictions = []
        self.references = []
        self.use_eos = self.args['lm']['use_eos']
        self.des = self.args['lm'].get('disable_eos_sampling', False)
        self.cutoff_lengths = []

        self.predictions_batch = []
        self.references_batch = []
        self.predictions_batch_idxs = []

    def update(self, batch_data):
        outputs_batch, targets_batch, target_lengths_batch, observations_batch, predicted_lengths_batch  = batch_data

        for output, target, target_length, pred_length in zip(outputs_batch, targets_batch, target_lengths_batch, predicted_lengths_batch):
            if target_length > 100 or len(output) <= 1 or len(target) <= 1:
                self.cutoff_lengths.append("NA")
                continue

            # find the best cutoff for bleu
            predictions = []
            for i in range(pred_length - 7, pred_length + 7):
                predictions.append(" ".join([self.data.target_vocab.idx2word[i.item()] for i in output[:i]]))
            if self.use_eos:
                target_length = target_length.item() - 1  # Don't use EOS in BLEU score
                if not self.des:
                    predictions = [
                            " ".join(predictions[-1].split()[:-1]) # get rid of final EOS
                            ]

            ref = [" ".join([self.data.target_vocab.idx2word[i.item()] for i in target[:target_length]])]

            self.predictions_batch.extend(predictions)
            self.predictions_batch_idxs.append(len(self.predictions_batch))
            self.references_batch.extend(ref)

            if len(self.references_batch) >= 1000: # arbitrary number - evaluate every
                self.evaluate_batches()
                self.clear_batches()

    def evaluate_batches(self):
        references = self.detokenize(self.references_batch)
        predictions = self.detokenize(self.predictions_batch)
        for ref, start_idx, end_idx in zip(
                                        references,
                                        [0] + self.predictions_batch_idxs[:-1],
                                        self.predictions_batch_idxs
                                        ):
            if start_idx != end_idx:
                max_idx = np.argmax(
                        np.array([sacrebleu.corpus_bleu(p, ref).score for p in predictions[start_idx:end_idx]])
                        )
                self.predictions.append(predictions[start_idx + max_idx])
                self.references.append(ref)
                self.cutoff_lengths.append(f"{len(predictions[start_idx:end_idx][max_idx])}")
            else:
                self.cutoff_lengths.append("NA")

    def clear_batches(self):
        self.predictions_batch = []
        self.references_batch = []
        self.predictions_batch_idxs = []

    def report(self):
        if self.references_batch:
            self.evaluate_batches()
            self.clear_batches()
        bleu = sacrebleu.corpus_bleu(self.predictions, [self.references])
        tqdm.write("Oracle bleu score: {}".format(bleu.score))
        outfile_name = os.path.join(get_results_dir_of_args(self.args), "oracle_bleu.{}".format(self.split_name))
        with open(outfile_name, "w") as f:
            try:
                f.write("{}\n{}\n".format(bleu.score, str(bleu)))
            except ZeroDivisionError:
                f.write("{}\nNA\n".format(bleu.score))

        outfile_name = os.path.join(get_results_dir_of_args(self.args), "oracle_bleu_lengths.{}".format(self.split_name))
        with open(outfile_name, "w") as f:
            f.write("\n".join(self.cutoff_lengths))


class OracleEndReporter(Reporter):
    """
    Evalutes exact match accuracy, truncating outputs where the gold targets end.

    If eos sampling is disabled, the targets will have an eos that is never
    predicted, so subtract one from target lengths before using them to
    truncate sequences. No eos training doesn't have this problem because
    eos tokens are not included in the targets.
    """
    def __init__(self, args, name, data, **kwargs):
        super().__init__(args, name, data)
        self.total = 0
        self.total_correct = 0

    def update(self, batch_data):
        outputs_batch, targets_batch, target_lengths_batch, observations_batch, _ = batch_data
        # see docstring for explanation
        if self.args['lm'].get('disable_eos_sampling', False):
            target_lengths_batch -= 1 
        for output, target, target_length in zip(outputs_batch, targets_batch, target_lengths_batch):
            if len(output) >= target_length and torch.all(torch.eq(output[:target_length], target[:target_length])):
                self.total_correct += 1
            self.total += 1
    
    def report(self):
        outfile_name = "{}/oracle_exact_match_acc.{}".format(get_results_dir_of_args(self.args), self.split_name)
        accuracy = self.total_correct / self.total
        tqdm.write("Exact match accuracy (Oracle ending) ({}): {} (total: {})".format(self.split_name, accuracy, self.total))
        with open(outfile_name, "w") as f:
            f.write("{}\n".format(accuracy))


class PerplexityReporter(Reporter):
    """
    Reports the perplexity of the model on some target dataset.
    """

    def __init__(self, args, name, *extra_args, **kwargs):
        self.args = args
        self.split_name = name
        self.ce_loss = 0
        self.total = 0

    def update(self, batch_data):
        logits_batch, targets_batch, lengths_batch, _ = batch_data
        batch_size, max_seq_len, vocab_size = logits_batch.size()

        for logits, target, length in zip(logits_batch, targets_batch, lengths_batch):
            self.ce_loss += F.cross_entropy(logits[:length, :], target[:length], reduction='sum')
            self.total += length.item() if isinstance(length, torch.Tensor) else length

    def report(self):
        outfile_name = '{}/{}.perplexity'.format(get_results_dir_of_args(self.args), self.split_name)
        tqdm.write("Loss: {}".format(self.ce_loss.cpu().item() / self.total))
        ppl = torch.exp(self.ce_loss.cpu() / self.total).item()
        with open(outfile_name, 'w') as outfile:
            outfile.write("{}\n".format(ppl))


class Seq2SeqSamplesReporter(Reporter):
    """
    Saves generated outputs along with their inputs and targets to disk.
    """

    def __init__(self, args, name, data, **kwargs):
        super().__init__(args, name, data)
        self.predictions = []
        self.targets = []
        self.inputs = []

    def update(self, batch_data):
        outputs_batch, targets_batch, target_lengths_batch, observations_batch, _ = batch_data
        for output, target, target_length, observation in zip(outputs_batch, targets_batch, target_lengths_batch, observations_batch):
            self.predictions.append(" ".join([self.data.target_vocab.idx2word[i.item()] for i in output]))
            self.targets.append(" ".join([self.data.target_vocab.idx2word[i.item()] for i in target[:target_length]]))
            self.inputs.append(" ".join(observation.source_tokens))

    def report(self):
        outfile_name = "{}/predictions.{}".format(get_results_dir_of_args(self.args), self.split_name)
        with open(outfile_name, 'w') as f:
            f.write("\n".join(self.predictions))
        outfile_name = "{}/targets.{}".format(get_results_dir_of_args(self.args), self.split_name)
        with open(outfile_name, 'w') as f:
            f.write("\n".join(self.targets))
        outfile_name = "{}/inputs.{}".format(get_results_dir_of_args(self.args), self.split_name)
        with open(outfile_name, 'w') as f:
            f.write("\n".join(self.inputs))





standard_reporter_lookup = {
        'perplexity': PerplexityReporter,
        'close_bracket': CloseBracketReporter, # for Dyck-m,k
        }

sampling_reporter_lookup = {
        'bleu': BleuReporter,
        'exact_match': ExactMatchReporter,
        'oracle_bleu': OracleBleuReporter,
        'oracle_eos': OracleEndReporter,
        'seq2seq_samples': Seq2SeqSamplesReporter,
        }

hidden_state_reporter_lookup = {
        'save_cache': CacheReporter, # used for plots
        }

def requires_updates(method):
    return method not in ['save_cache'] 
        

def record_training_hidden_states_lm(args, data, lm, methods):
    data_batches = data.get_train_dataloader(shuffle=False)
    reporters = [hidden_state_reporter_lookup[method](args, "train", data, lm)
            for method in methods if method in hidden_state_reporter_lookup]
    for seqs_batch, targets_batch, lengths_batch, ml_observations_batch in tqdm(data_batches, desc="[training]"):
        logits_batch = lm(seqs_batch, lengths_batch)
        for reporter in reporters:
            reporter.update((logits_batch, targets_batch, lengths_batch, ml_observations_batch))

    for reporter in reporters:
        reporter.report()

    lm.dump_cache(clear=True)

@torch.no_grad()
def evaluate_lm(args, lm, data, name, truncator=None):
    """
    Run evaluations for language models.
    """
    methods = []
    for method in args['reporter']['methods']:
        if isinstance(method, dict):
            methods.append(next(iter(method.keys())))
        else:
            methods.append(method)
    standard_reporters = []
    update_standard_reporters = False
    hidden_state_reporters = []
    update_hidden_state_reporters = False
    for method in methods:
        if method in standard_reporter_lookup:
            standard_reporters.append(standard_reporter_lookup[method](args, name, data))
            if requires_updates(method):
                update_standard_reporters = True
        if method in hidden_state_reporter_lookup:
            hidden_state_reporters.append(hidden_state_reporter_lookup[method](args, name, data, lm))
            update_hidden_state_reporters = True

    # if reporting hidden states, we need training data as well...
    # but only do this if we're evaluating the dev set
    if name == 'dev' and update_hidden_state_reporters:
        record_training_hidden_states_lm(args, data, lm, methods)

    if name == 'dev':
        data_batches = data.get_dev_dataloader(shuffle=False)
    elif name == 'test':
        data_batches = data.get_test_dataloader(shuffle=False)

    if update_standard_reporters or update_hidden_state_reporters:
        if truncator is not None:
            for (
                    seqs_batch,
                    (targets_batch_in, targets_batch),
                    (lengths_batch, _),
                    ml_observations_batch
                ) in tqdm(data_batches, desc="[{}-lm]".format(name)):
                logits_batch = lm(seqs_batch, lengths_batch)
                truncated_batch = truncator(seqs_batch, targets_batch_in, lengths_batch, lengths_batch, None, train=False)
                for reporter in standard_reporters + hidden_state_reporters:
                    reporter.update((logits_batch, targets_batch_in, truncated_batch, ml_observations_batch))
        else:
            for seqs_batch, targets_batch, lengths_batch, ml_observations_batch in tqdm(data_batches, desc="[{}]".format(name)):
                logits_batch = lm(seqs_batch, lengths_batch)
                for reporter in standard_reporters + hidden_state_reporters:
                    reporter.update((logits_batch, targets_batch, lengths_batch, ml_observations_batch))

    for reporter in standard_reporters + hidden_state_reporters:
        reporter.report()

    if update_hidden_state_reporters:
        lm.dump_cache(clear=True)

def record_training_hidden_states_seq2seq(args, data, lm, methods):
    data_batches = data.get_train_dataloader(shuffle=False)
    reporters = [hidden_state_reporter_lookup[method](args, "train", data, lm)
            for method in methods if method in hidden_state_reporter_lookup]
    if "plot_pca_sampled" in methods:
        reporters.append(sampling_reporter_lookup["plot_pca_sampled"](args, "train", data, lm))
    for (
            seqs_batch,
            (targets_batch_in, targets_batch_out),
            (input_length_batch, target_length_batch),
            observations_batch
        ) in tqdm(data_batches, desc="[train-seq2seq]"):
        logits_batch = lm(seqs_batch, targets_batch_in, input_length_batch, target_length_batch)
        for reporter in reporters:
            reporter.update((logits_batch, targets_batch_in, target_length_batch, observations_batch))

    for reporter in reporters:
        reporter.report()
    lm.dump_cache(clear=True)


@torch.no_grad()
def evaluate_seq2seq(args, lm, data, truncator, name):
    """
    Evaluates seq2seq models.

    We run two types of reporters: standard reporters input the target sequence
    into the model to get predicted probabilities of each token. (This is used
    in perplexity calculations for example). Sampling reporters greedily decode
    the best token at each time step, and are evaluated using exact match (for
    SCAN) or BLEU (for MT).

    Args:
        args (dict): arguments specified by the config yaml
        lm (models.Seq2Seq): the seq2seq model to evaluate
        data (data.Seq2SeqDataset): the dataset that contains the dev/test data
        truncator (truncation_models.TruncationModel): the model used to know
            where sampled sequences should end (for the non-eos and
            disabled-eos-sampling conditions.)
        name (str): indicates whether this is the dev or test evaluation.
            Used to label output files.
    """
    # Set up reporters
    methods = args['reporter']['methods']
    print(methods)
    lm.teacher_forcing_ratio = 1.0
    standard_reporters = []
    update_standard_reporters = False
    sampling_reporters = []
    update_sampling_reporters = False
    hidden_state_reporters = []
    update_hidden_state_reporters = False
    for method in methods:
        if method in standard_reporter_lookup:
            standard_reporters.append(standard_reporter_lookup[method](args, name, data))
            if requires_updates(method):
                update_standard_reporters = True
        if method in sampling_reporter_lookup:
            if requires_updates(method):
                update_sampling_reporters = True
            if method == "plot_pca_sampled":
                sampling_reporters.append(sampling_reporter_lookup[method](args, '{}_sampled'.format(name), data, lm))
            else:
                sampling_reporters.append(sampling_reporter_lookup[method](args, '{}_sampled'.format(name), data, truncator=truncator))

        if method in hidden_state_reporter_lookup:
            hidden_state_reporters.append(hidden_state_reporter_lookup[method](args, name, data, lm))
            update_hidden_state_reporters = True

    # if reporting hidden states, we need training data as well...
    # but only do this if we're evaluating the dev set
    if name == 'dev' and (update_hidden_state_reporters or "plot_pca_sampled" in methods):
        record_training_hidden_states_seq2seq(args, data, lm, methods)

    # load data
    if args['data']['dataset_type'] == "scan":
        torch.manual_seed(0)
    if name == 'dev':
        data_batches = data.get_dev_dataloader(shuffle=args['data']["dataset_type"]=="scan")
    elif name == 'test':
        data_batches = data.get_test_dataloader(shuffle=args['data']["dataset_type"]=="scan")

    pad_id = data.target_vocab.word2idx['<pad>']
    # Standard reporters: Run models with target outputs given.
    if update_standard_reporters or update_hidden_state_reporters:
        for (
                seqs_batch,
                (targets_batch_in, targets_batch_out),
                (input_length_batch, target_length_batch),
                observations_batch
            ) in tqdm(data_batches, desc="[{}-seq2seq]".format(name)):
            logits_batch = lm(seqs_batch, targets_batch_in, input_length_batch, target_length_batch)
            if logits_batch is None: continue
            truncation_preds_batch = truncator(seqs_batch,
                                               targets_batch_out,
                                               input_length_batch,
                                               target_length_batch,
                                               pad_id)
            # Update the internal states of the standard reporters
            for reporter in standard_reporters + hidden_state_reporters:
                reporter.update((logits_batch,
                                targets_batch_out,
                                target_length_batch,
                                observations_batch,
                                truncation_preds_batch))


    # Sampling reporters: Run greedy-decoding from models.
    if update_sampling_reporters:
        sos_id = data.target_vocab.word2idx['<sos>']
        eos_id = data.target_vocab.word2idx['<eos>']
        max_seq_len = args['lm'].get("max_seq_len", 50)
        for (
                seqs_batch, targets_batch, lengths_batch, observations_batch
            ) in tqdm(data_batches, desc="[{}-greedy-decoding]".format(name)):
            outputs_batch = lm.sample(seqs_batch,
                                      lengths_batch[0],
                                      sos_id=sos_id,
                                      eos_id=eos_id,
                                      max_length=max_seq_len)
            # Pad outputs so we can index into them
            outputs_batch_padded = nn.utils.rnn.pad_sequence(outputs_batch,
                                                             padding_value=pad_id,
                                                             batch_first=True)
            truncation_preds_batch = truncator(seqs_batch,
                                               outputs_batch_padded,
                                               lengths_batch[0],
                                               lengths_batch[1],
                                               pad_id)
            # Update the internal states of the sampling reporters
            for reporter in sampling_reporters:
                reporter.update((outputs_batch,
                                 targets_batch[1],
                                 lengths_batch[1],
                                 observations_batch,
                                 truncation_preds_batch))

    # Summarize the results from all of the reporters
    for reporter in standard_reporters + sampling_reporters + hidden_state_reporters:
        reporter.report()

    if update_hidden_state_reporters:
        lm.dump_cache(clear=True)

