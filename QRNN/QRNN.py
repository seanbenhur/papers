import torch
import torch.nn as nn
import torch.nn.functional as F


class QRNNLayer(nn.Module):
    def __init__(
        self,
        batch_size,
        input_size,
        n_filters,
        kernel_size,
        embed_size,
        device,
        dropout,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.conv1 = nn.Conv1d(self.input_size, self.n_filters, self.kernel_size)
        self.conv2 = nn.Conv1d(self.input_size, self.n_filters, self.kernel_size)
        self.conv3 = nn.Conv1d(self.input_size, self.n_filters, self.kernel_size)

    def forward(self, masked_input, h, c):
        Z, F, O = self.masked_conv(masked_input)
        h, c = self.pool(c, Z, F, O)
        masked_input = h
        return masked_input, h, c

    def masked_conv(self, x):
        pad = torch.zeros([self.batch_size, 1, self.input_size], device=self.device)
        x = torch.cat([pad, x], 1).permute(0, 2, 1)
        Z = torch.tanh((self.conv1(x)))
        F = torch.sigmoid((self.conv2(x)))
        O = torch.sigmoid((self.conv3(x)))
        one_mask = torch.ones_like(F, device=self.device) - F
        F = 1 - self.dropout(one_mask)
        return Z.permute(0, 2, 1), F.permute(0, 2, 1), O.permute(0, 2, 1)

    def pool(self, prev_c, Z, F, O):
        c = torch.mul(F, prev_c) + torch.mul(1 - F, Z)
        h = torch.mul(O, c)
        return h, c


class QRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        n_filters,
        kernel_size,
        batch_size,
        seq_len,
        n_layers,
        device,
        dropout,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(self.seq_len * self.n_filters, 1)
        self.qrnn_layers = nn.ModuleList(
            [
                QRNNLayer(
                    self.batch_size,
                    embed_size if self.n_layers == 0 else self.n_filters,
                    self.n_filters,
                    self.kernel_size,
                    self.embed_size,
                    self.device,
                    dropout,
                )
                for i in range(self.n_layers)
            ]
        )

        def forward(self, x, target):
            x = self.embedding(x)
            h = torch.zeros(
                [self.batch_size, self.seq_len, self.n_filters], device=self.device
            )
            c = torch.zeros_like(h, device=self.device)

            masked_input = x
            for l, layer in enumerate(self.qrnn_layers):
                masked_input, h, c = layer(masked_input, h, c)
            dense_input = h.view([self.batch_size, -1])
            logits = self.linear(dense_input)
            return logits
