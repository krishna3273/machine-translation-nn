import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.utils import extract_archive, download_from_url
from torchtext.vocab import Vocab
from collections import Counter
import io
from torch.utils.tensorboard import SummaryWriter
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from models_seq_to_seq import *
from utils import *


print("Loading data")
url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


print("Building Vocabulary")
def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update([token.lower() for token in tokenizer(string_)])
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq=2)


de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

print("Preprocessing the data")
def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor([de_vocab[token.lower()] for token in de_tokenizer(raw_de)],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token.lower()] for token in en_tokenizer(raw_en)],
                                  dtype=torch.long)
        data.append((de_tensor_, en_tensor_))
    return data


train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)

PAD_IDX_DE = de_vocab['<pad>']
BOS_IDX_DE = de_vocab['<bos>']
EOS_IDX_DE = de_vocab['<eos>']
PAD_IDX_EN = en_vocab['<pad>']
BOS_IDX_EN = en_vocab['<bos>']
EOS_IDX_EN = en_vocab['<eos>']

num_epochs = 20
learning_rate = 0.001
BATCH_SIZE = 64


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX_DE]), de_item, torch.tensor([EOS_IDX_DE])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX_EN]), en_item, torch.tensor([EOS_IDX_EN])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX_DE)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX_EN)
    return de_batch, en_batch


if __name__ == "__main__":
    print("Creating Batches with DataLoader")
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                           shuffle=True, collate_fn=generate_batch)

    print("Initialsing Models")
    load_model = False
    input_size_encoder = len(de_vocab)
    input_size_decoder = len(en_vocab)
    output_size = input_size_decoder
    embedding_size_encoder = 300
    embedding_size_decoder = 300
    hidden_size = 1024
    num_layers = 2
    dropout_param_encoder = 0.5
    dropout_param_decoder = 0.5

    writer = SummaryWriter(f'runs/seq2seq/loss_plot')
    step = 0
    encoder = Encoder(input_size_encoder, embedding_size_encoder, hidden_size, num_layers, dropout_param_encoder)
    decoder = Decoder(input_size_decoder, embedding_size_decoder, hidden_size, output_size, num_layers,
                      dropout_param_decoder)
    model = seq2seq(encoder, decoder)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_EN)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if load_model:
        load_checkpoint(torch.load('checkpoint.pth.tar'), model, optimizer)
    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."
    for i in range(num_epochs):
        print(f'epoch [{i + 1}/{num_epochs}]')
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

        model.eval()

        translated_sentence = translate_sentence(model, sentence, de_vocab, en_vocab)

        print(f"Translated example sentence: \n {translated_sentence}")

        model.train()

        for j, (de_batch, en_batch) in enumerate(train_iter):
            input_data = de_batch
            target_data = en_batch
            output = model(input_data, target_data)
            output = output[1:].reshape(-1, output.shape[2])
            target = target_data[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            print(f"batch_num={j}")
            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1
