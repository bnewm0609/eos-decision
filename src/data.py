from collections import Counter, namedtuple
import json
import jsonlines
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define named tuples for storing different kinds of data
MLTuple = namedtuple("MLTuple", ["good_sents", "good_verb_ids", "bad_verb_ids", "difference_idxs"])
WikiTuple = namedtuple("WikiTuple", ["token_idxs", "sentence", "tokens"])
SCANTuple = namedtuple("SCANTuple", ["source_idxs", "target_idxs",  "source_tokens", "target_tokens", "token_idxs", "target_length"])

class Dataset:
    """Based on:
        - https://github.com/salesforce/awd-lstm-lm/blob/master/data.py
        - https://github.com/john-hewitt/rnns-stacks-internal/blob/master/rnns-stacks/dataset.py
    """

    def __init__(self, args):
        self.args = args
        self.batch_size = args['lm']['batch_size']
        self.vocab = Vocab()
        generate_vocab = not self.vocab.load(args['data'].get('vocab_path', None))
        self.save_neighbor_idxs = args['data'].get('save_neighbor_idxs', False)
        if self.save_neighbor_idxs:
            self.neighbor_idx_map = []

        self.train_data = self.load_tokenized_data(args['data']['train_path'], 'train', generate_vocab)
        self.dev_data = self.load_tokenized_data(args['data']['dev_path'], 'dev')
        self.test_data = self.load_tokenized_data(args['data']['test_path'], 'test')
        tqdm.write('Vocabulary size: {}'.format(len(self.vocab)))
        args['data']['vocab_size'] = len(self.vocab) # I don't like to modify args, but it's appropriate here
        if generate_vocab:
            self.vocab.save(args['data']['vocab_path'])

    def load_tokenized_data(self, path, name, gen_vocab=False):
        if gen_vocab:
            for special_token in ['<unk>', '<pad>']:
                self.vocab.add_word(special_token)
        all_data = []
        total_tokens = 0
        unk_idx = self.vocab.word2idx['<unk>']
        with open(path, 'r') as data_file:
            for line in tqdm(data_file, desc="[loading {} data]".format(name)):
                tokens = line.split()
                if gen_vocab:
                    indices = [self.vocab.add_word(token) for token in tokens]
                else:
                    indices = [self.vocab.word2idx.get(token, unk_idx) for token in tokens]
                if len(indices) > 1:
                    token_idxs = [i + total_tokens for i in range(len(indices))]
                    total_tokens += len(token_idxs)
                    all_data.append(WikiTuple(token_idxs, indices, tokens))
        return all_data

    def custom_pad(self, observations):
        seqs = [torch.tensor(obs.sentence[:-1], device=self.args['device'], dtype=torch.long) for obs in observations]
        lengths = torch.tensor([len(x) for x in seqs], device=self.args['device'], dtype=torch.long)
        targets = [torch.tensor(obs.sentence[1:], device=self.args['device'], dtype=torch.long) for obs in observations]
        seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0) # 0 is the <eos> token, so this is ok
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

        if self.save_neighbor_idxs:
            for obs in observations:
                for idx in obs.token_idxs[:-1]: # follow seqs, bc that's what's being fed into neighbors
                    self.neighbor_idx_map.append(idx)

        return seqs, targets, lengths, observations 

    def get_train_dataloader(self, shuffle=True):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

    def get_dev_dataloader(self, shuffle=False):
        return DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

    def get_test_dataloader(self, shuffle=False):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)


class DatasetDyckEnding(Dataset):
    def __init__(self, args):
        """
        Technically this isn't a seq2seq dataset, but we might be able to
        shoehorn it in - the truth is that we need a way to get two sequences
        to our truncation model (one of inputs, the other of where the sequence
        can end.
        """
        super().__init__(args)
        self.target_vocab = self.vocab # to train truncation model we need the pad idx from the target vocab

    def process_line(self, line):
        line = line.strip()
        source_tkns = line.split()
        can_end = []
        stack = 0
        for token in source_tkns:
            if '(' in token:
                stack += 1
            else:
                stack -= 1

            can_end.append(1 if stack == 0 else 0)

        return source_tkns, can_end

    def load_tokenized_data(self, path, name, gen_vocab=False):
        if gen_vocab:
            for special_token in ['<unk>', '<pad>']:
                self.vocab.add_word(special_token)
        all_data = []
        total_tokens = 0
        unk_idx = self.vocab.word2idx['<unk>']
        with open(path, 'r') as data_file:
            for line in tqdm(data_file, desc="[loading {} data]".format(name)):
                tokens, can_end = self.process_line(line)
                if gen_vocab:
                    indices = [self.vocab.add_word(token) for token in tokens]
                else:
                    indices = [self.vocab.word2idx.get(token, unk_idx) for token in tokens]
                if len(indices) > 1:
                    token_idxs = [i + total_tokens for i in range(len(indices))]
                    total_tokens += len(token_idxs)
                    all_data.append(SCANTuple(indices, can_end, tokens, can_end, token_idxs, None))
        return all_data

    def custom_pad(self, observations):
        seqs = [torch.tensor(obs.source_idxs[:-1], device=self.args['device'], dtype=torch.long) for obs in observations]
        lengths = torch.tensor([len(x) for x in seqs], device=self.args['device'], dtype=torch.long)
        targets = [torch.tensor(obs.source_idxs[1:], device=self.args['device'], dtype=torch.long) for obs in observations]
        can_end = [torch.tensor(obs.target_idxs[:-1], device=self.args['device'], dtype=torch.float) for obs in observations]
        seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0) # 0 is the <eos> token, so this is ok
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
        can_end = nn.utils.rnn.pad_sequence(can_end, batch_first=True, padding_value=0)

        if self.save_neighbor_idxs:
            for obs in observations:
                for idx in obs.token_idxs[:-1]: # follow seqs, bc that's what's being fed into neighbors
                    self.neighbor_idx_map.append(idx)

        # return seqs, targets, lengths, observations 
        return seqs, (can_end, can_end), (lengths, lengths), observations 


class Seq2SeqDataset(Dataset):
    """Defines base class for seq2seq datasets."""

    def __init__(self, args):
        """
        Initializes class with vocab and data splits.
        
        Initializes the source and target vocabs and loads them from disk if
        they have already been created. Loads train, dev, and test data from
        disk as well.

        Args:
            args: A dict of arguments defined by the config yaml file.
        """
        self.args = args
        self.batch_size = args['lm']['batch_size']
        self.use_eos = args['lm']['use_eos']

        # prepare pre-created vocabularies
        self.source_vocab = Vocab()
        self.target_vocab = Vocab()
        self.generate_vocab = not self.source_vocab.load(args['data'].get('input_vocab_path', None))
        generate_vocab = not self.target_vocab.load(args['data'].get('target_vocab_path', None))

        # load in data splits
        self.train_data = self.load_tokenized_data(args['data']['train_path'], 'train', generate_vocab)
        self.dev_data = self.load_tokenized_data(args['data']['dev_path'], 'dev')
        self.test_data = self.load_tokenized_data(args['data']['test_path'], 'test')

        # log info about vocabularies and store for later
        tqdm.write('Source Vocabulary size: {}'.format(len(self.source_vocab)))
        tqdm.write('Target Vocabulary size: {}'.format(len(self.target_vocab)))
        args['data']['input_vocab_size'] = len(self.source_vocab) # I don't like to modify args, but it's appropriate here
        args['data']['target_vocab_size'] = len(self.target_vocab) # I don't like to modify args, but it's appropriate here

    def process_line(self, line):
        """
        Processes a line of data.

        One line of data should correspond to one example. It should contain
        source and target. Converting a line into the source and target should
        be dataset dependent.

        Args:
            line: string containing a line from a file. Likely new-line terminated.
        Returns:
            tuple(list(str), list(str)). The first list is source tokens and the
            second is the target tokens.
        """
        pass

    def load_tokenized_data(self, path, name, gen_vocab=False):
        """
        Loads space-tokenized data from disk into memory.

        The datafile should contain all tokens separated by whitespace as they
        are broken up by the `.split()` method. The return value will be of
        SCANTuple, for all seq2seq data. This method is called separately to
        obtain training, development, and testing data.

        Args:
            path: string containing the path to the data file.
            name: string containing the name of the data split. One of
                ('train', 'dev', 'test').
            gen_vocab: bool indicating whether the parsed tokens should be
                added to the vocabulary. True if constructing the vocabulary
                false otherwise.
        Returns:
            list(SCANTuple) containing all source and target tokens examples
            for the split along with some metadata about each example.
        """
        all_data = []
        total_tokens = 0

        # Add special tokens if we're building the vocab
        if gen_vocab:
            for special_token in ['<eos>', '<unk>', '<pad>']:
                self.source_vocab.add_word(special_token)
                self.target_vocab.add_word(special_token)
            self.target_vocab.add_word('<sos>')

        # Special tokens used if some token doesn't exist in vocab
        src_unk_idx = self.source_vocab.word2idx['<unk>']
        tgt_unk_idx = self.target_vocab.word2idx['<unk>']

        with open(path, "r") as data_file:
            for line in data_file:
                processed_line = self.process_line(line)
                if len(processed_line) == 3:
                    source_tkns, target_tkns, target_len = processed_line
                else:
                    source_tkns, target_tkns = processed_line
                    target_len = None

                # source tokens always get an eos token
                if source_tkns[-1] != '<eos>':
                    source_tkns.append('<eos>')
                # target tokens always begin with an sos token
                target_tkns = ['<sos>'] + target_tkns

                # whether we use eos is defined by the config
                if self.use_eos:
                    target_tkns += ['<eos>']

                # converts tokens to indices
                if gen_vocab:
                    source_idxs = [self.source_vocab.add_word(token) for token in source_tkns]
                    target_idxs = [self.target_vocab.add_word(token) for token in target_tkns]
                else:
                    source_idxs = [self.source_vocab.word2idx.get(token, src_unk_idx) for token in source_tkns]
                    target_idxs = [self.target_vocab.word2idx.get(token, tgt_unk_idx) for token in target_tkns]

                # tracks absolute index of tokens (can be used for linking
                # hidden states with tokens)
                token_idxs = [i + total_tokens for i in range(len(target_idxs))]
                total_tokens += len(token_idxs)
                all_data.append(SCANTuple(source_idxs, target_idxs, source_tkns, target_tkns, token_idxs, target_len))
        return all_data


    def custom_pad(self, observations):
        """
        Pads batches before sending them to models.

        The padding implemented here is pretty standard. We use packed tensors
        on the source side, so we need to reverse sort the source sequences by
        length. This method is called when iterating through a dataloader 
        obtained by calling get_{train,dev,test}_dataloader.

        Args:
            observations: list(SCANTuples). Length should be the size of a batch.
        Returns:
            A large tuple containing everything we could ever want to use for
            training: The source sequences, the, target sequences, the source
            and target lengths, as well as the observations themselves.
        """
        # Set up torch tensors of sequences along with their lengths
        seqs = [torch.tensor(obs.source_idxs, device=self.args['device'], dtype=torch.long) for obs in observations]
        source_lengths = [len(x) for x in seqs]
        targets_in = [torch.tensor(obs.target_idxs[:-1], device=self.args['device'], dtype=torch.long) for obs in observations]
        targets_out = [torch.tensor(obs.target_idxs[1:], device=self.args['device'], dtype=torch.long) for obs in observations]
        target_lengths = []
        for i, obs in enumerate(observations):
            if obs.target_length is None:
                # Note that 
                target_lengths.append(len(targets_in[i]))
            else:
                target_lengths.append(obs.target_length)

        # Add padding to sources and targets
        source_pad_value = self.source_vocab.word2idx['<pad>']
        target_pad_value = self.target_vocab.word2idx['<pad>']
        seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=source_pad_value) 
        targets_in = nn.utils.rnn.pad_sequence(targets_in, batch_first=True, padding_value=target_pad_value)
        targets_out = nn.utils.rnn.pad_sequence(targets_out, batch_first=True, padding_value=target_pad_value)

        # Now sort all fields by the lengths of the sources
        source_len_sort = np.argsort(-np.array(source_lengths))
        seqs = seqs[source_len_sort]
        targets_in = targets_in[source_len_sort]
        targets_out = targets_out[source_len_sort]
        source_lengths = torch.tensor(source_lengths, device=self.args['device'], dtype=torch.long)[source_len_sort]
        target_lengths = torch.tensor(target_lengths, device=self.args['device'], dtype=torch.long)[source_len_sort]
        # We need to sort the observations as well, which is easiest to do with
        # numpy, but numpy converts named tuples into np.arrays, which we don't
        # want, so the + [None] is hacky way to prevent this conversion.
        observations = np.array(observations + [None])[:-1][source_len_sort].tolist()

        return seqs, (targets_in, targets_out), (source_lengths, target_lengths), observations 


class DatasetSCAN(Seq2SeqDataset):
    def __init__(self, args):
        """
        Initializes DatasetSCAN object.

        For more information see superclass docstring.

        Args:
            args: A dict of arguments defined by the config yaml file.
        """
        super().__init__(args)
        if self.generate_vocab:
            self.source_vocab.save("data/scan/scan_input_vocab.json")
            self.target_vocab.save("data/scan/scan_target_vocab.json")

    def process_line(self, line):
        """
        Processes a single line from a data file.

        For the SCAN dataset, each source is prepended with "IN: " and each
        target is prepended with "OUT: ", so we split a line into source and
        target using the former.

        For more information see superclass docstring.

        Args:
            line: string containing a line from a file. Likely new-line terminated.
        Returns:
            tuple(list(str), list(str)). The first element is a list of source
            tokens and second is the target tokens.
        """
        line = line.strip()
        if "OUT:" in line:
            source, target = line.split(" OUT: ")
            source_tkns = source.split()[1:] # get rid of the "IN:" at the beginning
        else:
            target = line
            source_tkns = []
        target_tkns = target.split() 
        return source_tkns, target_tkns


class DatasetMT(Seq2SeqDataset):
    def __init__(self, args):
        """
        Initializes DatasetSCAN object.

        If a source and target vocab path were given, but not yet created,
        save the generated vocabularies in those paths, otherwise use the
        default ones. For more information see superclass docstring.

        Args:
            args: A dict of arguments defined by the config yaml file.
        """
        super().__init__(args)
        if self.generate_vocab:
            # determine where to store the vocabularies generated
            data_dir = os.path.dirname(args['data']['train_path'])
            source_vocab_file = self.args['data'].get('input_vocab_path', None)
            target_vocab_file = self.args['data'].get('target_vocab_path', None)
            if source_vocab_file is None or target_vocab_file is None:
                self.source_vocab.save(os.path.join(data_dir, "src_vocab.json"))
                self.target_vocab.save(os.path.join(data_dir, "tgt_vocab.json"))
            else:
                self.source_vocab.save(source_vocab_file)
                self.target_vocab.save(target_vocab_file)

    def process_line(self, line):
        """
        Processes a single line from a data file.

        For the MT dataset, the source and target are separated by a tab,
        so we split each line on the tab character. We also remove any
        trailing punctuation from the decoder side because trailing punctuation
        can serve as a signal for ending sentences. Right now, we do NOT convert
        words to lowercase.
        For more information see superclass docstring.

        Args:
            line: string containing a line from a file. Likely new-line terminated.
        Returns:
            tuple(list(str), list(str)). The first element is a list of source
            tokens and second is the target tokens.
        """
        line = line.strip()
        if "\t" in line:
            source, target = line.split("\t")
            source_tkns = source.split()
        else:
            target = line
            source_tkns = []
        target_tkns = target.split() 
        return source_tkns, target_tkns


class Vocab:
    """Based on https://github.com/salesforce/awd-lstm-lm/blob/master/data.py"""
    def __init__(self):
        """Initializes Vocab object with no state"""
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        """
        Adds a given word to the vocab object.

        If the word is not in the vocab, it is assigned a unique index based
        on how many words have previously been added. Regardless of the word's
        uniqueness, the total number of words and the count of the word is
        incremented. Here, word and token are synonymous.

        Args:
            word: string containing the token to add to the vocabulary.
        Returns:
            the unique index for this word 
        """
        if not word in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)

        idx = self.word2idx[word]
        self.counter[idx] += 1
        self.total += 1
        return idx

    def __len__(self):
        """Allows for calling `len` on a vocab object."""
        return len(self.word2idx)

    def save(self, path):
        """
        Saves the vocab object to the given path in json format.
        
        Note that word2idx is not saved because it can be reconstructed from
        idx2word.

        Args:
            path: string containing the path indicating where to save the vocab
        """
        output = {
                'idx2word': self.idx2word,
                'counter': self.counter,
                'total': self.total
                }
        
        with open(path, 'w') as outfile:
            outfile.write(json.dumps(output))

    def load(self, path):
        """
        Loads a vocab object from the given path.
        
        In addition to loading data from the specified path in json format,
        this method also reconstructs the word2idx object and stores it.

        Args:
            path: string containing the path to load vocab data from
        Returns:
            True if able to load vocab, False otherwise
        """
        if path is None or not os.path.exists(path):
            tqdm.write("Generating a new vocab")
            return False
        
        with open(path, 'r') as infile:
            lines = infile.read()
            vocab_dict = json.loads(lines)

        self.idx2word = vocab_dict['idx2word']
        self.word2idx = { w:i for i, w in enumerate(self.idx2word) }
        self.counter = vocab_dict['counter']
        self.total = vocab_dict['total']
        return True
