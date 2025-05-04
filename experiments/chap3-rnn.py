import random

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from rnn_utils import download_and_prepare_data, get_hyperparameters


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
def initialize_weights(model):
    # Loop through all named parameters in the model
    for name, param in model.named_parameters():
        # Check if the parameter has more than 1 dimension (e.g., weight matrices)
        if param.dim() > 1:
            # Use Xavier uniform initialization for weight matrices
            # This helps prevent vanishing/exploding gradients by keeping the variance constant
            nn.init.xavier_uniform_(param)
        else:
            # For 1D parameters (like biases), use simple uniform initialization
            nn.init.uniform_(param)


# %%
class ElmanRNNUnit(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.Uh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.Wh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.b = nn.Parameter(torch.randn(emb_dim))

    def forward(self, x, h):
        return torch.tanh(x @ self.Wh + h @ self.Uh + self.b)


class ElmanRNN(nn.Module):
    def __init__(self, emb_dim, num_layers):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.rnn_units = nn.ModuleList([ElmanRNNUnit(emb_dim) for _ in range(num_layers)])

    def forward(self, x):
        # x: 128 x 29 x 128
        # x is a batch of 128 "fragments" or "sequences"
        # each sequence has 29 tokens
        # each token is represented by a 128-dimensional embedding vector
        batch_size, seq_len, emb_dim = x.shape
        h_prev = [
            torch.zeros(batch_size, emb_dim, device=x.device) for _ in range(self.num_layers)
        ]
        # output: 29 x 128 x 128
        output = []
        # t = 0 ... 28
        for t in range(seq_len):
            # input_t: 128 x 128
            # input_t contains the embedding vectors for the t-th token of each sequence in the batch
            input_t = x[:, t]
            for l, rnn_unit in enumerate(self.rnn_units):
                # h_new: 128 x 128
                h_new = rnn_unit(input_t, h_prev[l])
                h_prev[l] = h_new
                # the result of the last rrn_unit is added to the output
                input_t = h_new
            # output contains the result of the last rnn_unit for each token in the sequence
            output.append(input_t)
        # ret: 128 x 29 x 128
        ret = torch.stack(output, dim=1)
        return ret


class RecurrentLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        # self.embedding will transform a tensor of indices into a vector of embeddings.
        # the valid indices are 0 ... vocab_size - 1
        self.rnn = ElmanRNN(emb_dim, num_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    # x: 128, 29
    def forward(self, x):
        # embedding.weights: 32011 x 128
        # embeddings: 128 x 29 x 128
        embeddings = self.embedding(x)
        rnn_output = self.rnn(embeddings)
        # rnn_output: 128 x 29 x 128
        # fc has a weight matrix of size 32011 x 128
        # the affine transformation is applied to the last dimension of the input tensor
        # i.e. output[b, s] = rnn_output[b, s] @ weights.T + bias
        # (for each batch and sequence)
        logits = self.fc(rnn_output)
        # logits: 128 x 29 x 32011
        return logits


# %%
device = torch.device("mps" if torch.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
vocab_size = len(tokenizer)
emb_dim, num_layers, batch_size, learning_rate, num_epochs = get_hyperparameters()
data_url = "https://www.thelmbook.com/data/news"
train_loader, test_loader = download_and_prepare_data(data_url, batch_size, tokenizer)
model = RecurrentLanguageModel(vocab_size, emb_dim, num_layers, tokenizer.pad_token_id)
initialize_weights(model)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    with open("loss.txt", "a") as f:
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                input_seq, target_seq = batch
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                batch_size_current, seq_len = input_seq.shape

                optimizer.zero_grad()
                # forward
                output = model(input_seq)
                output = output.reshape(batch_size_current * seq_len, vocab_size)
                target = target_seq.reshape(batch_size_current * seq_len)
                loss = criterion(output, target)
                f.write(f"{loss.item()}\n")
                loss.backward()
                optimizer.step()
