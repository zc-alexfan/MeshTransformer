import torch
import torch.nn as nn
import src.elytra.torch_utils as torch_utils


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        lm_emb_size,
        hidden_size,
        num_layers,
        bi_dir,
        act,
        dropout_p,
        output_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi_dir = bi_dir
        self.act = torch_utils.get_activation(act)
        if num_layers == 1:
            dropout_p = 0.0

        self.word_embeddings = nn.Embedding(
            len(pretrained_embeddings), lm_emb_size
        ).from_pretrained(torch.from_numpy(pretrained_embeddings), freeze=False)
        self.rnn = nn.LSTM(
            lm_emb_size,
            hidden_size,
            dropout=dropout_p,
            num_layers=num_layers,
            bidirectional=bi_dir,
        )
        lin_input_dim = hidden_size
        if bi_dir:
            lin_input_dim *= 2
        self.out = nn.Linear(lin_input_dim, output_size)

    def forward(self, inputs, hidden, batch_size):
        output = self.word_embeddings(inputs).view(1, batch_size, -1)
        output = self.act(output)
        output, hidden = self.rnn(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
