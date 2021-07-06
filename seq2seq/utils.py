import torch
from torchtext.data.utils import get_tokenizer



def translate_sentence(model, sentence, german, english, bos='<bos>',eos='<eos>', max_length=50):

    # Load german tokenizer
    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    tokens = [token.lower() for token in de_tokenizer(sentence)]


    # Add <BOS> and <EOS> in beginning and end respectively
    tokens.insert(0, bos)
    tokens.append(eos)

    # Go through each german token and convert to an index
    text_to_indices = [german.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.stoi[bos]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]])

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.stoi[eos]:
            break

    translated_sentence = [english.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]





def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
