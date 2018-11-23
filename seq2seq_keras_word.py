"""
author: Hoang Ho
This script is my attempt to rewrite a seq2seq model with pretrained glovemodel at word level 
for neural machine translation task.
I will apply this on English to German dataset taken from http://www.manythings.org/anki/

# Summary of the algorithm
- We start with English input sequences and use pretrained glove from https://nlp.stanford.edu/projects/glove/
    to obtain the embedding for input sequences.
- Similarly, we use https://www.spinningbytes.com/resources/wordembeddings/ for embedding target sequences.
- An encoder LSTM turns input sequences to 2 state vectors: the last LSTM hidden state and cell state
    (discarding the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence token <SOS>)
    - Feed the state vectors and 1-word target sequence
        to the decoder to produce predictions for the next word
    - Sample the next word using these predictions
        (we simply use argmax).
    - Append the sampled word to the target sequence
    - Repeat until we generate the end-of-sequence token <EOS> or we
        hit the sequence length limit.

reference: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Activation
from keras.activations import softmax as Softmax
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import unicodedata
import re

BATCH_SIZE = 128  # Batch size for training
NUM_SAMPLES = 12500
EPOCHS = 50
OPTIMIZER = "adam"
EMBED_DIM = 100
HIDDEN_DIM = 256
DATA_PATH = 'deu.txt'


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between words and punctation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sequence
    # so that the model know when to start and stop predicting
    w = "<SOS> " + w + " <EOS>"
    return w


eng_embedding = loadGloveModel('glove.6B.100d.txt')
ger_embedding = loadGloveModel('embed_tweets_de_100D_fasttext')

input_texts = []
target_texts = []
target_words = set()

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.read().split("\n")
    for line in lines[:NUM_SAMPLES]:
        input_text, target_text = line.split('\t')
        input_text = preprocess_sentence(input_text)
        target_text = preprocess_sentence(target_text)
        input_texts.append(input_text)
        target_texts.append(target_text)
        target_words.update(target_text.split())

target_words = sorted(list(target_words))

max_input_seqlen = max([len(txt.split()) for txt in input_texts])
max_target_seqlen = max([len(txt.split()) for txt in target_texts])
target_vocab_size = len(target_words) + 1

# get decoder input data
decoder_data = []
for text in target_texts:
    tmp = []
    for word in text.split():
        embed = np.random.randn(EMBED_DIM)
        if word in ger_embedding:
            embed = ger_embedding[word]
        tmp.append(embed)
    decoder_data.append(tmp)
decoder_data = pad_sequences(decoder_data, max_target_seqlen, padding="post")

# get decoder target data
targword2idx = dict([(word, i + 1) for i, word in enumerate(target_words)])
idx2targword = dict((i, word) for word, i in targword2idx.items())
decoder_target_data = []
for text in target_texts:
    tmp = []
    for idx, word in enumerate(text.split()):
        if idx > 0:
            tmp.append(targword2idx[word])
    decoder_target_data.append(tmp)
decoder_target_data = pad_sequences(
    decoder_target_data, max_target_seqlen, padding="post")
decoder_target_data = to_categorical(decoder_target_data)

# get encoder data
encoder_data = []
for text in input_texts:
    tmp = []
    for word in text.split():
        embed = np.random.randn(EMBED_DIM)
        if word in eng_embedding:
            embed = eng_embedding[word]
        tmp.append(embed)
    encoder_data.append(tmp)
encoder_data = pad_sequences(encoder_data, max_input_seqlen, padding="post")

# construct model
encoder_inputs = Input(shape=(max_input_seqlen, EMBED_DIM))
encoder_lstm = LSTM(HIDDEN_DIM, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, EMBED_DIM))
decoder_lstm = LSTM(HIDDEN_DIM, return_sequences=True,
                    return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(
    target_vocab_size, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# define training model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer=OPTIMIZER,
              loss='categorical_crossentropy', metrics=["acc"])
print(model.summary())
filename = 'seq2seq_keras.h5'
# checkpoint = ModelCheckpoint(
#     filename, verbose=1, save_best_only=True, mode='min')
checkpoint = ModelCheckpoint(filename, verbose=1, mode='min')
model.fit([encoder_data, decoder_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS, callbacks=[checkpoint], validation_split=0.2)


# create inference model
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(HIDDEN_DIM,))
decoder_state_input_c = Input(shape=(HIDDEN_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
# decoder_outputs = (BATCH_SIZE, seqlen, HIDDEN_DIM)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
# decoder_outputs = (BATCH_SIZE, seqlen, target_token_size)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


def decode(input_seq):
    states = encoder_model.predict(input_seq)
    target_seq = np.random.randn(EMBED_DIM)
    target_seq = [[target_seq]]
    stop_condition = False
    prediction = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states)
        sampled_token_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx2targword[sampled_token_idx]
        prediction += sampled_word + " "

        if (sampled_word == '<EOS>' or len(prediction) > max_target_seqlen):
            stop_condition = True

        if sampled_word in ger_embedding:
            target_seq = ger_embedding[sampled_word]
        else:
            target_seq = np.random.randn(EMBED_DIM)
        target_seq = [[target_seq]]
        states = [h, c]

    return prediction


actual, predicted = list(), list()

for index in range(100):
    input_seq = encoder_data[index]
    input_seq = np.expand_dims(input_seq, axis=0)
    actual.append(target_texts[index].split())
    prediction = decode(input_seq)
    predicted.append(prediction.split())
    print('-')
    print("Input sentence: ", input_texts[index])
    print("Translation: ", prediction)
    print("Truth value: ", target_texts[index])

print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' %
      corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' %
      corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' %
      corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def predict_sent(input_text):
    input_seq = []
    tmp = []
    text = preprocess_sentence(input_text)
    for word in text.split():
        embed = np.random.randn(EMBED_DIM)
        if word in eng_embedding:
            embed = eng_embedding[word]
        tmp.append(embed)
    input_seq.append(tmp)
    input_seq = pad_sequences(input_seq, max_input_seqlen, padding="post")
    states = encoder_model.predict(input_seq)
    target_seq = np.random.randn(EMBED_DIM)
    target_seq = [[target_seq]]
    target_text = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states)

        sampled_token_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx2targword[sampled_token_idx]
        target_text += sampled_word + " "

        if sampled_word == '<EOS>' or len(target_text) >= max_target_seqlen:
            stop_condition = True

        if sampled_word in ger_embedding:
            target_seq = ger_embedding[sampled_word]
        else:
            target_seq = np.random.randn(EMBED_DIM)
        target_seq = [[target_seq]]

        states = [h, c]
    return target_text.strip()


predict_sent("She is beautiful")
