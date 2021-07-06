import torch
import torch.nn as nn
import torch.optim as optim
import random


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_param):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_param)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_param)

    def forward(self, x):
        # x_shape:(seq_length,batch_size)
        embedding = self.dropout(self.embedding(x))
        # embedding_shape:(seq_length,batch_size,embedding_size)
        outputs, (hidden_state, cell_state) = self.rnn(embedding)
        return hidden_state, cell_state


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_param):
        super(Decoder, self).__init__()
        self.input_size=input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_param)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_param)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x_shape:(batch_size),we want (1,batch_size)
        x = x.unsqueeze(0)
        # x_shape:(1,batch_size)
        embedding = self.dropout(self.embedding(x))
        # embedding_shape:(1,batch_size,embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs_shape:(1,batch_size,hidden_size)
        predictions = self.fc(outputs)
        # predictions_shape:(1,batch_size,output_size)
        return predictions.squeeze(0), hidden, cell


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = self.decoder.input_size

    def forward(self, source, target, actual_predict_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        hidden, cell = self.encoder(source)
        x = target[0]
        outputs = torch.zeros(target_len, batch_size, self.target_vocab_size)
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < actual_predict_ratio else best_guess
        return outputs
