import torch
import torch.nn as nn
import numpy as np
from src.elytra.rnn_encoder import LSTMEncoder
from src.elytra.rnn_decoder import LSTMDecoder
import src.elytra.rnn_utils as rnn_utils


class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        word2idx,
        hidden_size,
        num_layers=1,
        bi_dir=False,
        dropout=0.0,
        pad_index=0,
        teacher_forcinig_ratio=1.0,
        sos_token="<sos>",
        eos_token="<eos>",
        max_num_tokens=30,
        act="relu",
    ):
        super().__init__()
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)
        self.hidden_size = hidden_size
        embeddings = pretrained_embeddings
        self.encoder = LSTMEncoder(
            embeddings, embeddings.shape[1], hidden_size, num_layers, bi_dir, dropout
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.decoder = LSTMDecoder(
            embeddings,
            embeddings.shape[1],
            hidden_size,
            num_layers,
            bi_dir,
            act,
            dropout,
            self.vocab_size,
        )
        self.teacher_forcinig_ratio = teacher_forcinig_ratio
        self.idx2word = [k for k, _ in self.word2idx.items()]
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_num_tokens = max_num_tokens

    def encoder_forward(self, batch):
        """
        Take a batch of inputs and return last hidden state of encoder.
        Inputs:
            batch['packed_phrases']
                The packed version of the above phrases.
        """
        packed_phrases = batch["packed_phrases"]
        batch_size = batch["packed_phrases"].batch_sizes.max().item()
        DEVICE = batch["packed_phrases"].data.device

        # encoder forward
        h0c0 = self.encoder.initHiddenCell(batch_size, DEVICE)
        _, hidden = self.encoder(h0c0, packed_phrases)

        return {"hidden": hidden}

    def inference(self, batch):
        """
        Take a batch of inputs and return the reconstructed seq.
        Inputs:
            batch['packed_phrases']
                The packed version of the above phrases.
            Note: both inputs are assumed sorted in descending order.

        """
        batch_size = batch["packed_phrases"].batch_sizes.max().item()
        DEVICE = batch["packed_phrases"].data.device

        out_dict = self.encoder_forward(batch)
        hidden = out_dict["hidden"]

        # decoder forward
        inputs = torch.LongTensor(
            [[self.word2idx[self.sos_token]] for _ in range(batch_size)]
        )
        inputs = inputs.to(DEVICE)
        sample_sent = [self.sos_token]
        for step_i in range(self.max_num_tokens):
            output, hidden = self.decoder(inputs, hidden, batch_size)
            # use its own prediction
            _, inputs = output.max(dim=1)
            sample_sent.append(self.idx2word[inputs.item()])
            if inputs.item() == self.word2idx[self.eos_token]:
                break
        out_dict["recon_seq"] = sample_sent
        return out_dict

    def forward(self, batch):
        """
        Take a batch of inputs and return the loss.
        Inputs:
            batch['target_phrase_indices']:
                A list of Long Tensor, each is a phrase.
            batch['packed_phrases']
                The packed version of the above phrases.
            Note: both inputs are assumed sorted in descending order.

        """
        batch_size = batch["packed_phrases"].batch_sizes.max().item()
        DEVICE = batch["packed_phrases"].data.device

        target_phrase_indices = batch["target_phrase_indices"]
        assert isinstance(target_phrase_indices, list)
        out_dict = self.encoder_forward(batch)
        hidden = out_dict["hidden"]
        padded_target = rnn_utils.pad_phrase_indices(target_phrase_indices)
        padded_target = padded_target[:, 1:].to(DEVICE)  # remove <sos>

        # decoder forward
        inputs = torch.LongTensor(
            [[self.word2idx[self.sos_token]] for _ in range(batch_size)]
        )
        inputs = inputs.to(DEVICE)
        total_loss = 0.0
        num_steps = padded_target.size(1)
        for step_i in range(num_steps):
            curr_target = padded_target[:, step_i]
            output, hidden = self.decoder(inputs, hidden, batch_size)
            if np.random.rand() < self.teacher_forcinig_ratio:
                # teacher forcing
                inputs = curr_target.unsqueeze(1)
            else:
                # use its own prediction
                _, inputs = output.max(dim=1)

            loss = self.criterion(output, curr_target)
            total_loss += loss
        total_loss /= num_steps
        out_dict["loss"] = total_loss
        return out_dict
