import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        lm_emb_size,
        hidden_size,
        num_layers,
        bi_dir,
        dropout_p,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(
            len(pretrained_embeddings), lm_emb_size
        ).from_pretrained(torch.from_numpy(pretrained_embeddings), freeze=False)

        if num_layers == 1:
            dropout_p = 0.0

        self.lstm = nn.LSTM(
            input_size=lm_emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=bi_dir,
            dropout=dropout_p,
        )
        self.lstm.num_bi_layer = 2 if self.lstm.bidirectional else 1

    def forward(self, h0c0, ph_input):
        """
        ph_input is a packed sequence of word indices
        each example in a batch is a (img, query) pair
        e.g. batch size 64 means there are 64 such pairs
        """
        padded = rnn.pad_packed_sequence(ph_input, batch_first=True, padding_value=0)

        embedded = self.word_embeddings(padded[0])

        packed_embedded = rnn.pack_padded_sequence(
            embedded, padded[1], batch_first=True
        )
        outputs, hidden = self.lstm(packed_embedded, h0c0)
        return outputs, hidden

    def initHidden(self, batch_size):
        return torch.zeros(
            self.lstm.num_bi_layer * self.lstm.num_layers, batch_size, self.hidden_size
        )

    def initCell(self, batch_size):
        return torch.zeros(
            self.lstm.num_bi_layer * self.lstm.num_layers, batch_size, self.hidden_size
        )

    def initHiddenCell(self, batch_size, dev):
        c0 = self.initCell(batch_size).to(dev)
        h0 = self.initHidden(batch_size).to(dev)
        h0c0 = (h0, c0)
        return h0c0
