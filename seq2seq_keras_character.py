"""
author: Hoang Ho
This script is my attempt to rewrite a seq2seq model with pretrained glovemodel at character level 
for neural machine translation task.
I will apply this on English to German dataset taken from http://www.manythings.org/anki/

reference: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Activation
from keras.activations import softmax as Softmax
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from keras import backend as K
import numpy as np
import unicodedata
import re

BATCH_SIZE = 128  # Batch size for training
NUM_SAMPLES = 12500
EPOCHS = 30
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
    w = "\t " + w + " \n"
    return w


eng_embedding = loadGloveModel('glove.6B.100d.txt')

input_texts = []
target_texts = []
target_chars = set()

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.read().split("\n")
    for line in lines[:NUM_SAMPLES]:
        input_text, target_text = line.split('\t')
        input_text = preprocess_sentence(input_text)
        target_text = preprocess_sentence(target_text)
        input_texts.append(input_text)
        target_texts.append(target_text)
        target_chars.update(list(target_text))

target_chars = sorted(list(target_chars))

# get attributes from data
max_input_seqlen = max([len(txt.split()) for txt in input_texts])
max_target_seqlen = max([len(txt) for txt in target_texts])
target_token_size = len(target_chars)

# get decoder data
targchars2idx = dict([(char, i) for i, char in enumerate(target_chars)])
idx2targchars = dict((i, char) for char, i in targchars2idx.items())
decoder_data = np.zeros(
    shape=(NUM_SAMPLES, max_target_seqlen, target_token_size))
decoder_target_data = np.zeros(
    shape=(NUM_SAMPLES, max_target_seqlen, target_token_size))

for textIdx, text in enumerate(target_texts):
    for idx, char in enumerate(text):
        c2idx = targchars2idx[char]
        decoder_data[textIdx, idx, c2idx] = 1
        if idx > 0:
            decoder_target_data[textIdx, idx - 1, c2idx] = 1

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

decoder_inputs = Input(shape=(None, target_token_size))
decoder_lstm = LSTM(HIDDEN_DIM, return_sequences=True,
                    return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(
    target_token_size, activation="softmax", name="decoder_dense")
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

# construct inference models
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
    target_seq = np.zeros((1, 1, target_token_size))
    target_seq[0, 0, targchars2idx['\t']] = 1
    stop_condition = False
    prediction = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states)
        sampled_token_idx = np.argmax(output_tokens[0, -1, :])
        sampled_char = idx2targchars[sampled_token_idx]
        prediction += sampled_char

        if (sampled_char == '\n' or len(prediction) > max_target_seqlen):
            stop_condition = True

        target_seq = np.zeros((1, 1, target_token_size))
        target_seq[0, 0, sampled_token_idx] = 1
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
    target_seq = np.zeros((1, 1, target_token_size))
    target_seq[0, 0, targchars2idx['\t']] = 1
    target_text = ''
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states)

        sampled_token_idx = np.argmax(output_tokens[0, -1, :])
        sampled_char = idx2targchars[sampled_token_idx]
        target_text += sampled_char

        if sampled_char == '\n' or len(target_text) >= max_target_seqlen:
            stop_condition = True

        target_seq = np.zeros((1, 1, target_token_size))
        target_seq[0, 0, sampled_token_idx] = 1

        states = [h, c]
    return target_text.strip()


predict_sent("She is beautiful")
