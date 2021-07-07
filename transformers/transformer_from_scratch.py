import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        assert (self.head_size * num_heads == embedding_size), "Embedding size must be divisible by number of heads"

        self.queries_fc = nn.Linear(self.head_size, self.head_size, bias=False)
        self.values_fc = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys_fc = nn.Linear(self.head_size, self.head_size, bias=False)
        self.out_fc = nn.Linear(embedding_size, embedding_size)

    def forward(self, values, keys, queries, mask):
        num_samples = values.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(num_samples, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(num_samples, key_len, self.num_heads, self.head_size)
        queries = queries.reshape(num_samples, query_len, self.num_heads, self.head_size)

        values = self.values_fc(values)
        keys = self.keys_fc(keys)
        queries = self.queries_fc(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e28"))

        attention = torch.softmax(energy / (self.embedding_size ** 0.5), 3)

        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(num_samples, query_len, self.embedding_size)

        return self.out_fc(out)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, num_heads)

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.hidden_size = forward_expansion * embedding_size

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, embedding_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        y = self.feed_forward(x)
        return self.dropout(self.norm2(x + y))


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, device,
                 max_length):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embedding_size, num_heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        num_samples, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(num_samples, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, num_heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_block = TransformerBlock(embedding_size, num_heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, forward_expansion, dropout, device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embedding_size, num_heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])
        self.out_fc = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        num_samples, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(num_samples, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.out_fc(out)
        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embedding_size=512, num_layers=6,
                 forward_expansion=4, num_heads=8, dropout=0, device="cpu", max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, num_layers,
                               num_heads, forward_expansion, dropout, device, max_length)
        self.decoder = Decoder(trg_vocab_size, embedding_size, num_layers,
                               num_heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)

    def make_trg_mask(self, trg):
        num_samples, trg_len = trg.shape
        return torch.tril(torch.ones((trg_len, trg_len))).expand(num_samples, 1, trg_len, trg_len).to(device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src, src_mask)
        return self.decoder(trg, enc_out, src_mask, trg_mask)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)
