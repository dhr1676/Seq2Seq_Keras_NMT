# Seq2Seq_Keras_NMT

In this Repo, I attempted to create a basic Seq2Seq model for Neural Machine Translation task applied on English to German translation dataset from http://www.manythings.org/anki/

There are two models available: a character-level model and word-level model. Both make use of glove pretrained model obtained from https://nlp.stanford.edu/projects/glove/ for English word embedding.  The word-level model also makes use of https://www.spinningbytes.com/resources/wordembeddings/ for German word embedding. In both cases, I'm using 100-dimension word vector embedding.

### Reference 
The script is an extension from: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py


