#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:43:32 2020

@author: daniele.gentili
"""


import pandas as pd
import numpy as np
import itertools
from collections import Counter
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from gensim.models import Word2Vec
import gensim.models.keyedvectors as key_vec
import re

def substitute(word):
    """
    Regular expressions to change words
    """
    word = word.lower()
    word = re.sub(r"[^A-Za-z0-9^,!./'+-=]", " ", word)
    word = re.sub(r"what's", "what is ", word)
    word = re.sub(r"'s", " ", word)
    word = re.sub(r"'ve", " have ", word)
    word = re.sub(r"n't", " not ", word)
    word = re.sub(r"i'm", "i am ", word)
    word = re.sub(r"'re", " are ", word)
    word = re.sub(r"all'", " a ", word)
    word = re.sub(r"dell'", " di ", word)
    word = re.sub(r"l'", " ", word)
    word = re.sub(r"'d", " would ", word)
    word = re.sub(r"-", " - ", word)
    word = re.sub(r"'ll", " will ", word)
    word = re.sub(r",", " ", word)
    word = re.sub(r"\'", " ", word)
    word = re.sub(r"\.", " ", word)
    word = re.sub(r"!", " ! ", word)
    word = re.sub(r":", " : ", word)
    word = re.sub(r" e g ", " eg ", word)
    
    return word

def clean(sentence, stem=False):
    """
    Appling regular expressions, and stemming if desired
    """
    sw = set(stopwords.words("english")) 
    if stem:
        stemmer = nltk.stem.SnowballStemmer("english")
    sentence = sentence.split()
    sentence = [substitute(w) for w in sentence]
    sentence = " ".join(sentence)
    sentence = word_tokenize(sentence)
    if stem: 
        sentence = [stemmer.stem(w) for w in sentence if w not in sw]
    else:
        sentence = [w for w in sentence if w not in sw]
    return sentence

    
def pad_sentences(sentences, padding_word="<PAD/>", num=37):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        if len(new_sentence) > num:
            new_sentence = new_sentence[:num]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def preprocess_data(sentences, labels):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]



#def w2v_preprocessing(sentences, lang=str):
#    en_model = key_vec.KeyedVectors.load_word2vec_format("resources/GoogleNews-vectors-negative300.bin", binary=True)
#    vocab = set(model.wv.vocab)
#    sentences = [[model.wv[w] for w in s if w in vocab] for s in sentences]
#    #sentences = [[np.mean(vex, axis=0)] for vex in sentences]
#        
#    return sentences, model
    