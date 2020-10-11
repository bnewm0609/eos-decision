"""Implements various sequence models"""
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm

import utils

class LanguageModel(nn.Module):
    """Base class for langauge models"""
    def __init__(self, args):
        """
        Initializes a language model.
        
        Args:
            args: A dict of arguments defined by the config yaml file.
        """
        super().__init__()
        self.args = args
        self.init_model()
        num_params = sum(p.numel() for p in self.parameters())
        tqdm.write(f"Model has {num_params} parameters")
        self.to(args['device'])
        self._record_hidden_states = False
        self.cache = []

    def init_model(self):
        """Sets up torch layers that make up the model."""
        pass

    def forward(self, seq):
        pass

    def load_pretrained_model(self, model_path=None):
        """
        Loads pretrained model weights into the current model.

        Args:
            model_path: optional string containing the path to load the model
                from. Default is generated from the config yaml file.
        """
        if model_path is None:
            model_path = "{}/model.params".format(utils.get_lm_path_of_args(self.args))
        tqdm.write("Loading model from {}".format(model_path))

        state_dict = torch.load(model_path, map_location=self.args['device'])
        self.load_state_dict(state_dict)
        self.to(self.args['device'])

    def generate_masks(self, input_lens):
        """
        Generates integer mask for sequences of given lengths.

        The mask is used for computing attention. A value of 0 means attention
        should be computed at that position, and a value of 1 means attention
        should not be computed for that position (e.g. because the token at
        that position is a padding token).

        Args:
            input_lens: torch.Tensor(batch_size) containing the lengths of the
                inputs sequences of a batch.
        Returns:
            torch.Tensor(batch_size, max_input_len). A mask tensor whose values
            are described above.
        """
        masks = torch.zeros((len(input_lens), input_lens.max())).to(self.args['device'])
        for i in range(len(input_lens)):
            masks[i, input_lens[i]:] = 1
        return masks

    def record_hidden_states(self):
        self._record_hidden_states = True

    def dump_cache(self, clear):
        if clear:
            self.cache = []
            return
        return np.vstack(self.cache)

class RNNLM(LanguageModel):

    def init_model(self):
        dropout_rate = self.args['lm']['dropout']
        vocab_size = self.args['data']['vocab_size']
        embed_dim = self.args['lm']['embed_dim']
        hidden_dim = self.args['lm']['hidden_dim']

        self.embed = nn.Embedding(vocab_size, embed_dim)

        # set up lm
        cell_type = self.args['lm']['cell_type']
        if cell_type in ('LSTM', 'standard'):
            pt_cell_class = nn.LSTM
        elif cell_type == 'GRU': 
            pt_cell_class = nn.GRU
        num_layers = self.args['lm']['num_layers']
        self.lm = pt_cell_class(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, lengths):
        embeddings = self.embed(inputs)
        hidden_states, _ = self.lm(embeddings)
        outputs = self.vocab_proj(self.dropout(hidden_states))

        if self._record_hidden_states:
            masks = ~self.generate_masks(lengths).bool()
            self.cache.extend(hidden_states[masks].cpu().numpy())

        return outputs


class Seq2Seq(nn.Module):
    """Base class for sequence-to-sequence models."""

    def __init__(self, args):
        """
        Initializes a seq2seq model.
        
        Args:
            args: A dict of arguments defined by the config yaml file.
        """
        super().__init__()
        self.args = args

        # sampling should be the same as setting this ratio to 1!
        self.teacher_forcing_ratio = args['lm'].get('teacher_forcing_ratio', 1.0)
        self.disable_eos = self.args['lm'].get('disable_eos_sampling', False)
        self.init_model()
        num_params = sum(p.numel() for p in self.parameters())
        tqdm.write(f"Model has {num_params} parameters")
        self.to(args['device'])
        self._record_hidden_states = False
        self.cache = []

    def init_model(self):
        """Sets up torch layers that make up the model."""
        pass

    def generate_masks(self, input_lens):
        """
        Generates integer mask for sequences of given lengths.

        The mask is used for computing attention. A value of 0 means attention
        should be computed at that position, and a value of 1 means attention
        should not be computed for that position (e.g. because the token at
        that position is a padding token).

        Args:
            input_lens: torch.Tensor(batch_size) containing the lengths of the
                inputs sequences of a batch.
        Returns:
            torch.Tensor(batch_size, max_input_len). A mask tensor whose values
            are described above.
        """
        masks = torch.zeros((len(input_lens), input_lens.max())).to(self.args['device'])
        for i in range(len(input_lens)):
            masks[i, input_lens[i]:] = 1
        return masks

    def load_pretrained_model(self, model_path=None):
        """
        Loads pretrained model weights into the current model.

        Args:
            model_path: optional string containing the path to load the model
                from. Default is generated from the config yaml file.
        """
        if model_path is None:
            model_path = "{}/model.params".format(utils.get_lm_path_of_args(self.args))
        tqdm.write("Loading model from {}".format(model_path))

        state_dict = torch.load(model_path, map_location=self.args['device'])
        self.load_state_dict(state_dict)
        self.to(self.args['device'])

    def encode(self, sources, source_lens, hiddens=None):
        """Encodes the source sentences. See subclass for details"""
        pass

    def decode(self, targets, dec_states, enc_states=None, enc_mask=None):
        """Decode the source sentences. See subclass for details"""
        pass

    def sample(self, sources, source_lens, hiddens=None, sos_id=1, eos_id=1, max_length=50):
        """
        Greedily decodes a target sequence for each source sequence.

        Even though the method is called "sample", we're actually doing greedy
        decoding.

        Args:
            sources: torch.Tensor(batch_size, max_seq_src_len) source sequences
            source_lens: torch.Tensor(batch_size) containing the length of each
                source sequence.
            hiddens: torch.Tensor(batch_size, max_seq_src_len, hidden_dim)
                containing optional intial states for the encoder.
            sos_id: int containing the id of the start of sequence token in
                the target vocab.
            eos_id: int containing the id of the end of sequence token in the
                target vocab.
            max_length: int containing the maximum sequence length.
        Returns:
            List(torch.Tensor(sequence length)) containing model outputs
        """
        batch_size, _ = sources.shape

        # Encode the sources
        dec_states, enc_hiddens, enc_mask = self.encode(sources, source_lens)

        # Prepare data structures for doing greedy decoding
        dec_input_t = torch.tensor([sos_id] * batch_size, device=sources.device).unsqueeze(1)
        samples = torch.empty((batch_size, max_length), device=sources.device, dtype=torch.long)
        sample_results = [None for _ in range(batch_size)]

        for t in range(max_length):
            # Take a decoding step
            logits, dec_output, dec_states = self.decode(dec_input_t, dec_states, enc_hiddens, enc_mask) 

            if self.disable_eos:
                # If exp(eos_id) = 0, eos won't be chosen at decoding time
                logits[..., eos_id] = -float("inf")

            # Greedily choose next token and store it
            max_idxs = torch.argmax(logits, dim=-1)
            samples[:, t] = max_idxs.squeeze(-1)

            # If we chose the eos token, we're done - store the sequence that
            # terminated in a tensor.
            for i in range(batch_size):
                if sample_results[i] is None and max_idxs[i].item() == eos_id:
                    sample_results[i] = samples[i, : t + 1]
            dec_input_t = max_idxs

        # Add any sequences that haven't terminated yet to the results tensor 
        for i in range(batch_size):
            if sample_results[i] is None:
                sample_results[i] = samples[i]

        return sample_results

    def record_hidden_states(self):
        self._record_hidden_states = True

    def dump_cache(self, clear):
        if clear:
            self.cache = []
        else:
            return np.vstack(self.cache)


class Seq2SeqLSTM(Seq2Seq):
    """Seq2Seq model that uses RNNs as encoders and decoders."""

    def __init__(self, args):
        super().__init__(args)

    def init_model(self):
        """Initialize archictecture based on config."""
        dropout_rate = self.args['lm']['dropout'] # should be 0.5
        source_vocab_size = self.args['data']['input_vocab_size']
        target_vocab_size = self.args['data']['target_vocab_size']
        embed_dim = self.args['lm']['embed_dim']
        hidden_dim = self.args['lm']['hidden_dim']
        self.bidirectional = self.args['lm'].get('bidirectional', False)
        self.use_attention = self.args['lm'].get('attention', False)
        if self.bidirectional:
            self.bidi_proj_cell = nn.Linear(2 * hidden_dim, hidden_dim)
            self.bidi_proj_hidden = nn.Linear(2 * hidden_dim, hidden_dim)

        if self.use_attention:
            if self.bidirectional:
                self.attn_proj = nn.Linear(2 * hidden_dim, hidden_dim)
                self.combined_attention_proj = nn.Linear(3 * hidden_dim, hidden_dim)
            else:
                self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
                self.combined_attention_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.source_embed = nn.Embedding(source_vocab_size, embed_dim)
        self.target_embed = nn.Embedding(target_vocab_size, embed_dim)

        # encoder
        self.encoder = self.init_encdec(embed_dim, hidden_dim, "encoder", bidirectional=self.bidirectional)

        # decoder
        self.decoder = self.init_encdec(embed_dim, hidden_dim, "decoder")
        self.target_vocab_proj = nn.Linear(hidden_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def init_encdec(self, embed_dim, hidden_dim, name, **kwargs):
        """
        Initialzes the encoder or decoder LSTM.

        Even though LSTMs natively support multiple layers, here we use
        nn.ModuleLists to simulate them for historical reasons. 

        Args:
            embed_dim: int specifying the dimension of embeddings
            hidden_dim: int specifying the dimension of the  hidden/cell vectors
            name: string, either 'encoder' or 'decoder' indicating which half
                of the model is being initialized.
            **kwargs: other keyword args that are passed to the LSTM constructor
        Returns:
            nn.ModuleList containing the rnn layers that makde up the encoder
            or decoder.
        """
        cell_type = self.args['lm'][f'{name}_cell_type']
        if cell_type == 'LSTM' or cell_type == 'standard':
            pt_cell_class = nn.LSTM
        # yes, even though class is Seq2SeqLSTM, we can use GRU's too
        elif cell_type == 'GRU': 
            pt_cell_class = nn.GRU

        num_layers = self.args['lm'][f'{name}_num_layers']
        setattr(self, f'{name}_num_layers', num_layers)

        # prepare  hidden dim sizes
        lstm_dim_map = [(embed_dim, hidden_dim)]
        for i in range(1, num_layers):
            lstm_dim_map.append((hidden_dim, hidden_dim))

        # Initialize the RNNs
        rnns = []
        for i in range(num_layers):
            rnns.append(pt_cell_class(*lstm_dim_map[i], batch_first=True, **kwargs))

        return nn.ModuleList(rnns)

    def forward(self, sources, targets, source_lens, target_lens, hiddens=None):
        """
        Runs the forward pass of the entire model.
        Args:
            sources: torch.Tensor(batch_size, max_src_seq_len) containing source sequences
            targets: torch.tensor(batch_size, max_tgt_seq_len) containing target sequecnes
            source_lens: torch.tensor(batch_size) containing the lengths of the source seqs
            target_lens: torch.tensor(batch_size) containing the lengths of the target seqs
            hiddens: (optional) list(torch.tensor(batch_size, 1, hidden_dim)) initial 
                hidden states to start each layer of the model with. This is unused.
        Returns:
            torch.Tensor(batch_size, max_src_seq_len, target_vocab_size)
            containing unnormalized log-probabilities (logits) for each token in 
            each sequence.
        """
        # encoder
        dec_states, enc_hiddens, enc_mask = self.encode(sources, source_lens)

        # decoder
        if self.use_attention:
            logits = self.decode(targets, dec_states, enc_hiddens, enc_mask, target_lens=target_lens)[0]
        else:
            logits = self.decode(targets, dec_states, target_lens=target_lens)[0] # for now, this will do

        return logits

    def encode(self, sources, source_lens, hiddens=None):
        """
        Runs the forward pass of the encoder.

        Args:
            sources: torch.Tensor(batch_size, max_src_seq_len) containing source sequences
            source_lens: torch.tensor(batch_size) containing the lengths of the source seqs
            hiddens: (optional) list(torch.tensor(batch_size, 1, hidden_dim)) initial 
                hidden states to start each layer of the model with. This is unused.
        Returns:
            A tuple containing three objects
                - The initial hidden states for the decoder
                - The padded encoder hidden states (when using attention)
                - Integer masks for valid source sentences (when using attention).
                  See superclass self.generate_masks documentation

            Type:
            tuple(
                tuple(torch.Tensor(batch_size, 1, hidden_dim),
                      torch.Tensor(batch_size, 1, hidden_dim)),
                torch.Tensor(batch_size, max_src_seq_len, hidden_dim),
                torch.Tensor(batch_size, max_src_seq_len)
            )
        """
        source_embeds = self.dropout(self.source_embed(sources))

        # Run encoder LSTM
        source_embeds = nn.utils.rnn.pack_padded_sequence(source_embeds, source_lens, batch_first=True)
        for i in range(self.encoder_num_layers):
            source_embeds, hc_f = self.encoder[i](source_embeds)
        enc_hiddens, dec_states = source_embeds, hc_f

        # For a bidirectional encoder, project the two hidden/cell states down
        # to the size of one.
        if self.bidirectional:
            dec_init_cell = self.bidi_proj_cell(torch.cat((dec_states[0][0], dec_states[0][1]), dim=1)).unsqueeze(0)
            dec_init_hidden = self.bidi_proj_cell(torch.cat((dec_states[1][0], dec_states[1][1]), dim=1)).unsqueeze(0)
            dec_states = (dec_init_cell, dec_init_hidden)

        # If using attention, we need to return length  masks and padded encoder
        # outputs at each time step.
        enc_mask = None
        if self.use_attention:
            enc_hiddens, _ = nn.utils.rnn.pad_packed_sequence(enc_hiddens)
            enc_hiddens = enc_hiddens.transpose(0, 1)
            enc_mask = self.generate_masks(source_lens)

        dec_states = self.prepare_decoder_hs(dec_states)
        return dec_states, enc_hiddens, enc_mask

    def decode(self, targets, dec_states, enc_states=None, enc_mask=None, target_lens=None):
        """
        Runs the forward pass of the decoder.

        With probability `self.teacher_forcing_ratio` the input to the
        decoder is greedily chosen based on the model's predictions. 
        Otherwise, the true target sequence is used.

        If we're using attention, first run the decoder normally and use
        those outputs to form decoder outputs.

        Args:
            targets: torch.tensor(batch_size, max_tgt_seq_len) containing target sequecnes
            dec_states: tuple(torch.tensor(batch_size, 1, hidden_dim), torch.tensor(batch_size, 1, hiden_dim))
                containing the intial hidden state for the decoder.
            enc_states: torch.Tensor(batch_size, max_src_seq_len, hidden_dim) LSTM hidden states
                from each time step during encoding time. (Optional - used with attention)
            enc_mask: torch.Tensor(batch_size, max_src_seq_len) source masks used for computing
                attention. (Optional - used with attention)
        Returns:
            A tuple containing three items:
                - logits for each token int decoder outputs at each time step
                  torch.Tensor(batch_size, max_tgt_seq_len, tgt_vocab_size)
                - decoder hidden states at each time step 
                - final decoder hidden and cell states
        """
        # encode targets
        targets = self.dropout(self.target_embed(targets))

        teacher_forcing = np.random.random() < self.teacher_forcing_ratio
        if teacher_forcing:
            # feed in entire target sequence batched
            new_dec_states = []
            for i in range(self.decoder_num_layers):
                targets, new_dec_state = self.decoder[i](targets, dec_states[i])
                new_dec_states.append(new_dec_state)
            dec_outputs = self.dropout(targets)
            if self.use_attention:
                dec_outputs = self.dropout(self.decode_attn(dec_outputs, enc_states, enc_mask))
        else:
            # greedily decode one token at a time
            dec_outputs = []
            seq_len = targets.size(1)
            target_t = targets[:, 0, :].unsqueeze(1)
            for i in range(seq_len):
                # decode a single step (decoder could have multiple layers)
                new_dec_states = []
                for i in range(self.decoder_num_layers):
                    target_t, new_dec_state = self.decoder[i](target_t, dec_states[i])
                    new_dec_states.append(new_dec_state)
                dec_states = new_dec_states
                if self.use_attention:
                    target_t = self.decode_attn(target_t, enc_states, enc_mask)
                target_t = self.dropout(target_t)
                dec_outputs.append(target_t)

                # greedily choose the next token
                top_i = torch.argmax(self.target_vocab_proj(target_t), dim=2)
                target_t = self.dropout(self.target_embed(top_i.detach()))
            dec_outputs = torch.cat(dec_outputs, dim=1)

        logits = self.target_vocab_proj(dec_outputs)

        if self._record_hidden_states and target_lens is not None:
            masks = ~self.generate_masks(target_lens).bool()
            self.cache.extend(dec_outputs[masks].cpu().numpy())
        elif self._record_hidden_states:
            self.cache.extend(dec_outputs.reshape(-1, dec_outputs.shape[-1]).cpu().numpy())

        return logits, dec_outputs, new_dec_states

    def decode_attn(self, dec_outputs, enc_states, enc_mask):
        """
        Runs the attention mechanism of the decoder if required by config.

        Uses dot product attention to compute weighted sum of encoder
        hidden states.

        Args:
            dec_outputs: The hidden states output by the decoder RNN.
            enc_states: The padded encoder hidden states (when using attention).
            enc_mask: Integer masks for valid source sentences (when using attention).
        Returns:
            torch.Tensor(dec_outputs.size()), containing sum of
            encoder hidden states weighted by attention weights.
        """
        # apply attention:
        enc_states_proj = self.attn_proj(enc_states)
        e_t = torch.bmm(enc_states_proj, dec_outputs.transpose(1, 2))

        # fill e_t with -inf in mask
        e_t.data.masked_fill_(enc_mask.unsqueeze(2).bool(), -float('inf'))

        # compute attention weights and calculate weighted sum
        alpha_ts = nn.functional.softmax(e_t, dim=1)
        a_ts = torch.bmm(alpha_ts.transpose(1, 2), enc_states) 
        output = torch.cat((dec_outputs, a_ts), dim=2)
        output = self.combined_attention_proj(output)
        output = torch.tanh(output)

        return output

    def prepare_decoder_hs(self, dec_states):
        """Repeats inital decoder states for each layer in the decoder."""
        return [dec_states for _ in range(self.decoder_num_layers)]


class Seq2SeqPrecomputed(Seq2Seq):
    """
    Dummy Seq2Seq model that maps inputs to outputs by parsing a file
    
    We might want to do this if we train a model with OpenNMT but want
    to run some custom evaluations on it
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.dev_path = args['lm']['dev_output_path']
        self.test_path = args['lm']['test_output_path']
        self.use_eos = args['lm']['use_eos']
        self.des = args['lm'].get('disable_eos_sampling', False)
        self.lookup = {}
        self.data = None

    def load_from_data(self, data):
        """
        Loads our "model" given a data.Seq2SeqDataset object.

        This method is called from `run.py`. Note that we're not construcing
        the data loader but instead are using the raw list(data.SCANTuple).
        """
        self.data = data
        self.lookup = self.create_lookup(data.dev_data, self.dev_path)
        self.lookup.update(self.create_lookup(data.test_data, self.test_path))

    def create_lookup(self, data_list, path):
        """
        Loads a and stores precomputed model output from path.

        The data the model should output is stored at `path`. We assume it's
        in the same order as in the `data_list`, which we obtain from the
        `data.Seq2SeqDataset`. Note that the file the `path` points to should
        only contain the tokenized model output.
        
        Args:
            data_list: list(data.SCANTuple) that contains the input data that
                will be produced
            path: a string containing the path to the file where the model
                output we load in is.
        """
        table = {}
        unk_idx = self.data.target_vocab.word2idx['<unk>'] 
        extra_token = []
        if self.use_eos and not self.des:
            extra_token.append(self.data.target_vocab.word2idx['<eos>'])

        with open(path) as df:
            for i, line in enumerate(df):
                if i == len(data_list):
                    print("Stopping early when creating model: found more outputs than in data. Ensure data files and output files match.")
                    break
                data = data_list[i].source_idxs.copy()
                key = pickle.dumps(data) # use pickle to serialize the source sequence

                processed_line = self.data.process_line(line)
                if len(processed_line) == 3:
                    source_tkns, target_tkns, target_len = processed_line
                else:
                    source_tkns, target_tkns = processed_line
                    target_len = None

                value = torch.tensor([
                    self.data.target_vocab.word2idx.get(token, unk_idx) for token in target_tkns] + extra_token
                    ).to(self.args['device'])
                table[key] = value
        return table

    def forward(self, sources, targets, source_lens, target_lens):
        """
        Runs the forward pass of the model.

        Note that this only works if the batch size is 1 because otherwise
        the padding tokens prevent us from finding the correct target sequence.
        """
        result_batch = []
        # this for loop should only run once
        for i, src_seq in enumerate(sources):
            key = pickle.dumps(src_seq[:source_lens[i]].tolist())
            result_batch.append(self.lookup[key])
        return result_batch

    def sample(self, sources, source_lens, hiddens=None, sos_id=1, eos_id=1, max_length=50):
        """
        Looks up the model outputs associated with the sequences listed in `sources`.
        """
        return self.forward(sources, None, source_lens, None)

    def load_pretrained_model(self, model_path):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class Seq2SeqPrecomputedHiddenStates(Seq2Seq):
    """
    Dummy Seq2Seq model that maps inputs to outputs by parsing a file
    
    We might want to do this if we train a model with OpenNMT but want
    to run some custom evaluations on it
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.caches = {
            'train': self.create_lookup('train'),
            'dev': self.create_lookup('valid'),
            'test': self.create_lookup('test'),
        }

        self.use_eos = args['lm']['use_eos']
        self.des = args['lm'].get('disable_eos_sampling', False)

        self.data_splits = iter(['train', 'dev', 'test'])
        self.current_split = next(self.data_splits)
        self.current_counter = 0

        self.data = None
        self.dataset_lengths = None

    def load_from_data(self, data):
        """
        Loads our "model" given a data.Seq2SeqDataset object.

        This method is called from `run.py`. Note that we're not construcing
        the data loader but instead are using the raw list(data.SCANTuple).
        """
        self.data = data
        self.dataset_lengths = {
                "train": len(self.data.train_data),
                "dev": len(self.data.dev_data),
                "test": len(self.data.test_data),
        }

    def slices_to_ints(self, slices_list):
        arr_len = sum([s.stop - s.start for s in slices_list])
        arr = np.empty(arr_len).astype(int)
        arr_ind = 0
        for s in slices_list:
            s_len = s.stop - s.start
            arr[arr_ind: arr_ind + s_len] = np.arange(s.start, s.stop)
            arr_ind += s_len
        return arr

    def create_lookup(self, split):
        """
        Loads a and stores precomputed model output from path.

        The data the model should output is stored at `path`. We assume it's
        in the same order as in the `data_list`, which we obtain from the
        `data.Seq2SeqDataset`. Note that the file the `path` points to should
        only contain the tokenized model output.
        
        Args:
            data_list: list(data.SCANTuple) that contains the input data that
                will be produced
            path: a string containing the path to the file where the model
                output we load in is.
        """

        cache_path = os.path.join(self.args['reporter']['results_path'], f"cache.{split}.npy")
        cache_map_path = os.path.join(self.args['reporter']['results_path'], f"cache_idxs.{split}.csv")
        cache = np.load(cache_path)

        cache_map_entries = [tuple(map(int, entry.split(','))) for entry in open(cache_map_path).readlines()]

        
        cache_map = {}
        total_length = 0
        for idx, length in cache_map_entries:
            cache_map[idx] = (slice(total_length, total_length + length), length)
            total_length += length

        # reorder the cache according to the sorted keys of cache_map
        sorted_slices = []
        sorted_lens = []
        sorted_idxs = []
        for idx in sorted(cache_map.keys()):
            sorted_idxs.append(idx)
            slce, length = cache_map[idx]
            sorted_slices.append(slce)
            sorted_lens.append(length)

        slice_ints = self.slices_to_ints(sorted_slices)
        cache = cache[slice_ints]

        # recreate the cache_map now that everything has been sorted
        cache_map = {}
        total_length = 0
        for idx, length in zip(sorted_idxs, sorted_lens):
            cache_map[idx] = (slice(total_length, total_length + length), length)
            total_length += length

        return cache, cache_map

    def dump_cache(self, clear=False):
        del clear
        return self.caches[self.current_split][0]

    def forward(self, sources, targets, source_lens, target_lens):
        """
        Runs the forward pass of the model.

        Note that this only works if the batch size is 1 because otherwise
        the padding tokens prevent us from finding the correct target sequence.
        """
        assert targets.shape[0] == 1 # batch size must be one

        if self.current_counter == self.dataset_lengths[self.current_split]:
            self.current_counter = 0
            try:
                self.current_split = next(self.data_splits)
            except:
                print("no more splits")

        cache, cache_map = self.caches[self.current_split]
        hs_slice, sent_length = cache_map.get(self.current_counter, (None, target_lens[0].cpu().item()))
        assert target_lens[0] == sent_length

        self.current_counter += 1

        return torch.from_numpy(cache[hs_slice]).unsqueeze(0) if hs_slice else None


    def sample(self, sources, source_lens, hiddens=None, sos_id=1, eos_id=1, max_length=50):
        """
        Looks up the model outputs associated with the sequences listed in `sources`.
        """
        raise ValueError("Cannot sample from pre-loaded hidden states")

    def load_pretrained_model(self, model_path):
        pass

    def train(self):
        pass

    def eval(self):
        pass
