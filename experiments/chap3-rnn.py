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
    # emb_dim: 128
    def __init__(self, emb_dim: int):
        super().__init__()
        # Uh: 128 x 128
        self.Uh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        # Wh: 128 x 128
        self.Wh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        # b: 128
        self.b = nn.Parameter(torch.randn(emb_dim))

    # x: 128 x 128, h: 128 x 128
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ self.Wh + h @ self.Uh + self.b)


class ElmanRNN(nn.Module):
    # emb_dim: 128, num_layers: 2
    def __init__(self, emb_dim: int, num_layers: int):
        super().__init__()
        # emb_dim: 128
        self.emb_dim = emb_dim
        # num_layers: 2
        self.num_layers = num_layers
        # rnn_units: [ElmanRNNUnit(128), ElmanRNNUnit(128)]
        self.rnn_units = nn.ModuleList([ElmanRNNUnit(emb_dim) for _ in range(num_layers)])

    def forward(self, embedded_batch: torch.Tensor) -> torch.Tensor:
        # embedded_batch: 128 x 29 x 128
        # batch_size: embedded_batch is a batch of 128 "fragments" or "sequences"
        # seq_len: each sequence has 29 tokens
        # emb_dim: each token is represented by a 128-dimensional embedding vector
        batch_size, seq_len, emb_dim = embedded_batch.shape
        h_prev = [
            torch.zeros(batch_size, emb_dim, device=embedded_batch.device) for _ in range(self.num_layers)
        ]
        # output: 29 x 128 x 128
        output = []
        # t = 0 ... 28
        # each t corresponds to a token in the sequence
        for t in range(seq_len):
            # input_t: 128 x 128
            # input_t contains the embedding vectors for the t-th token of each sequence in the batch
            input_t = embedded_batch[:, t]
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
    # vocab_size: 32011, emb_dim: 128, num_layers: 2
    def __init__(self, vocab_size, emb_dim, num_layers, pad_idx):
        super().__init__()
        # embedding.weights: 32011 x 128
        # each row of the embedding matrix corresponds to a token in the vocabulary
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = ElmanRNN(emb_dim, num_layers)
        # (* x 128) --fc--> (* x 32011)
        self.fc = nn.Linear(emb_dim, vocab_size)

    # batch_in: 128 x 29
    # batch_in is a batch of 128 sequences
    def forward(self, batch_in):
        # embedding.weights: 32011 x 128
        # embedded_batch: 128 x 29 x 128
        embedded_batch = self.embedding(batch_in)
        # 128 x 29 --embedding--> 128 x 29 x 128
        #       ^^                      ^^^^^^^^
        # self.embedding will transform a tensor of indices into a vector of embeddings.
        # the valid indices are 0 ... 32010
        rnn_output = self.rnn(embedded_batch)
        # rnn_output: 128 x 29 x 128
        # fc has a weight matrix of size 32011 x 128
        # the affine transformation is applied to the last dimension of the input tensor
        # i.e. output[b, s] = rnn_output[b, s] @ weights.T + bias
        # (for each batch and sequence)
        # in other words, it transforms each embedding vector v of size 128 into a vector l of size 32011
        # the element of l with the highest value corresponds to the most likely token following v and the context
        logits = self.fc(rnn_output)
        # logits: 128 x 29 x 32011
        return logits


# %%
device = torch.device("mps" if torch.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
# vocab_size: 32011
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
            # batch: (128 x 29, 128 x 29)
            for batch in train_loader:
                input_seq, target_seq = batch
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                # 128, 29
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
