import os, re, sys, json, csv, string, gzip
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pickle
import nltk
import random

import joblib

from itertools import chain
import logging

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin

import itertools

import logging
from optparse import OptionParser
import sys
from time import time

import argparse

DIGITS = list('1234567890')
UPPERCASE = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
LOWERCASE = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.lower())
PUNCT = list(",.?!:;’\"")
SPECIAL_CHARS = list("<>%|{}[]/\@#˜+-*=$ˆ&_()’")
TAB = '\t'


CODE_ROOT = '/h/vkpriya/pdnc-lrec/code'

def load_function_words():
    with open(os.path.join(CODE_ROOT, 'lexicons', 'func_words.txt'), 'r') as f:
        lines = f.readlines()
    lines = [x.split()[0] for x in lines]
    return lines

FUNC_WORDS = load_function_words()

class DummyVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_names, key=''):
#         self.featMat = featMat
        self.key = key
        self.feat_names = feat_names
        
    def transform(self, x, y=None):
        # x = [eval(x1) for x1 in x]
        return np.array(x.tolist())
    
    def fit(self, x, y=None):
        # x = [eval(x1) for x1 in x]
        return self
    
    def fit_transform(self, x, y=None):
        # x = [eval(x1) for x1 in x]
        return np.array(x.tolist())
    
    def get_feature_names(self):
        return self.feat_names

class OrdinalVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_names, key=''):
#         self.featMat = featMat
        self.key = key
        self.feat_names = feat_names
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
    def transform(self, x, y=None):
        return self.encoder.transform(np.array(x).reshape(-1,1))
    
    def fit(self, x, y=None):
        return self.encoder.fit(np.array(x).reshape(-1,1))
    
    def fit_transform(self, x, y=None):
        return self.encoder.fit_transform(np.array(x).reshape(-1,1))
    
    def get_feature_names(self):
        return self.feat_names


count_vocab = LOWERCASE + SPECIAL_CHARS + PUNCT
ratio_vocab = [None, DIGITS, UPPERCASE+LOWERCASE, UPPERCASE, TAB]
char_feats = []
for v in count_vocab:
    char_feats.append('COUNT_'+ v)
char_feats.extend(['COUNT_N', 'RATIO_DIGITS', 'RATIO_CHARS', 'RATIO_UPPER', 'RATIO_TAB'])

def tokenize(sent):
    sents = nltk.sent_tokenize(sent)
    words = [nltk.word_tokenize(x) for x in sents]
    return words

class charVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, count_vocab, ratio_vocab, feat_names):
        self.count_vocab = count_vocab 
        self.ratio_vocab = ratio_vocab
        self.feat_names = feat_names
    
    def preprocess(self, x):
        new_x = [' '.join(list(chain.from_iterable(x_))) for x_ in tqdm(x)]
        return new_x 

    def get_ratio_feats(self, counter):

        all_ = np.sum([counter[x] for x in counter])
        feats = []

        for vocab in self.ratio_vocab:
            if vocab is not None:
                sum_ = np.sum([counter[x] for x in vocab])
                feats.append(sum_/all_)
            else:
                feats.append(all_)

        return feats
    
    def get_counter(self, feat, counter):
        if feat.islower():
            return counter[feat]+counter[feat.upper()]
        else:
            return counter[feat]

    def get_feats(self, x):
        feats1 = np.zeros((len(x), len(self.count_vocab)))
        feats2 = np.zeros((len(x), len(self.ratio_vocab)))

        for ind, doc in tqdm(enumerate(x)):
            counter = Counter(list(doc))
            d_feat = [self.get_counter(f, counter) for f in self.count_vocab]
            feats1[ind] = d_feat
            feats2[ind] = self.get_ratio_feats(counter)
        
        feats = np.concatenate([feats1, feats2], axis=1)
        return feats 

    def transform(self, x):
        new_x = self.preprocess(x)
        feats = self.get_feats(new_x)
        return feats
    
    def fit(self, x):
        return self 

    def get_feature_names(self):
        return self.feat_names

class LexicalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def tokenize(self, t):
        if isinstance(t, str):
            sents = nltk.sent_tokenize(t)
            words = [nltk.word_tokenize(x) for x in sents]
            return words
        return t    
    
    def len_feats(self, words):
        sent_counts = []
        word_counts = []
        for sent in words:
            sent_counts.append(len(' '.join(sent)))
            for word in sent:
                word_counts.append(len(word))
        sent_len =  np.mean(sent_counts)

        counter = Counter(word_counts)
        short_ratio = sum([counter[x] for x in range(4)])/len(word_counts) 
        ratios = [counter[x]/len(word_counts) for x in range(4, 21)]
        feats = [sent_len, len(word_counts), np.mean(word_counts), short_ratio]
        feats.extend(ratios)

        return feats
    
    def fit(self,x,y=None):
        return self
    
    def transform(self,x,y=None):
        feats = [self.len_feats(t) for t in tqdm(x)]
        return feats
    
    def get_feature_names(self):
        feats = ['AVG_SENT_LEN', 'TOKEN_COUNT', 'AVG_WORD_LEN', 'SHORT_RATIO']
        for i in range(4, 21):
            feats.append('LEN_RATIO_'+str(i))
        return feats

def get_lexicon_feats(tokenized, feature_names, w2i, vecs, word2counts=None):
    words = [x.lower() for sublist in tokenized for x in sublist]
    
    feats = np.zeros(len(feature_names))
    counts = np.zeros(len(feature_names))
    
    for w in words:
        if w in w2i:
            feats = np.add(feats, vecs[w2i[w]])
            for ind, val in enumerate(vecs[w2i[w]]):
                if val != 0.0:
                    counts[ind] += 1
                    if word2counts:
                        word2counts[feature_names[ind]][w] += 1
    
    vec = []
    for x, y in zip(feats, counts):
        if y!=0:
            vec.append(x/y)
        else:
            vec.append(0)
    
    return vec

class LexiconVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, feat_names, w2i, vecs, key = ''):
        self.feat_names = feat_names
        self.w2i = w2i
        self.vecs = vecs
        self.key = key
    
    def transform(self, x, y=None):
        feats = [get_lexicon_feats(x_, self.feat_names, self.w2i, self.vecs) for x_ in x]
        return feats
    
    def fit(self, x, y=None):
        return self
    
    def get_feature_names(self):
        return self.feat_names


style_lexicon = pickle.load(open(os.path.join(CODE_ROOT, 'lexicons/style_lexicon.dict.pkl'), 'rb'))
style_names = style_lexicon['names']
style_vecs = style_lexicon['vecs']
style_w2i = style_lexicon['w2i']

nrc_lexicon = pickle.load(open(os.path.join(CODE_ROOT, 'lexicons/nrc_feats.dict.pkl'), 'rb'))
nrc_names = nrc_lexicon['names']
nrc_vecs = nrc_lexicon['vecs']
nrc_w2i = nrc_lexicon['w2i']

# FEATURE SET:

# Quote:
# - Stylo #input: tokenized
# - NRC lex, Style lex #input: tokenized
# - TfIdf #input: text

# Mentions:
# - TfIdf #input: text

# Ref Exp:
# - TfIdf #input: text

class CorpusFeatureBuilder:
    def __init__(self, traindf, testdf):
        self.traindf = traindf
        self.testdf = testdf
        
        self.charVect = charVectorizer(count_vocab, ratio_vocab, char_feats)
        self.funcVect = CountVectorizer(vocabulary=FUNC_WORDS)
        self.lenVect = LexicalFeatures()
        self.nrcVect = LexiconVectorizer(nrc_names, nrc_w2i, nrc_vecs)
        self.styleVect = LexiconVectorizer(style_names, style_w2i, style_vecs)
        
        self.quoteVect = TfidfVectorizer(stop_words=FUNC_WORDS)
        self.menVect = TfidfVectorizer()
        self.refVect = TfidfVectorizer()
    
    
    def fit_features(self):
        traindf = self.traindf

        quotes = traindf['qText'].tolist()
        tokenized_quotes = [tokenize(q) for q in quotes]

        mentions = [str(x) for x in traindf['menStr'].tolist()]
        
        refexps = [str(x) for x in traindf['refExp'].tolist()]
        
        stylo1 = self.charVect.fit_transform(tokenized_quotes)
        stylo2 = self.funcVect.fit_transform(quotes)
        stylo3 = self.lenVect.fit_transform(tokenized_quotes)
        
        nrclex = self.nrcVect.fit_transform(tokenized_quotes)
        stylelex = self.styleVect.fit_transform(tokenized_quotes)
        
        qtf = self.quoteVect.fit_transform(quotes)
        
        mtf = self.menVect.fit_transform(mentions)
        
        rtf = self.refVect.fit_transform(refexps)
        
        feats = np.concatenate([stylo1, stylo2.toarray(), stylo3, nrclex, stylelex, qtf.toarray(), mtf.toarray(), rtf.toarray()], axis=1)
        
        return feats
    
    def transform_features(self, testdf = None):
        if testdf is None:
            testdf = self.testdf
        
        quotes = testdf['qText'].tolist()
        tokenized_quotes = [tokenize(q) for q in quotes]

        mentions = [str(x) for x in testdf['menStr'].tolist()]
        
        refexps = [str(x) for x in testdf['refExp'].tolist()]
        
        stylo1 = self.charVect.transform(tokenized_quotes)
        stylo2 = self.funcVect.transform(quotes)
        stylo3 = self.lenVect.transform(tokenized_quotes)
        
        nrclex = self.nrcVect.transform(tokenized_quotes)
        stylelex = self.styleVect.transform(tokenized_quotes)
        
        qtf = self.quoteVect.transform(quotes)
        
        mtf = self.menVect.transform(mentions)
        
        rtf = self.refVect.transform(refexps)
        
        feats = np.concatenate([stylo1, stylo2.toarray(), stylo3, nrclex, stylelex, qtf.toarray(), mtf.toarray(), rtf.toarray()], axis=1)
        
        return feats
    
    def get_feature_names(self):
        feature_names = []
        
        for v, vname in zip([self.charVect, self.funcVect, self.lenVect, self.nrcVect, self.styleVect, self.quoteVect,  \
            self.menVect, self.refVect], ['STYLO', 'STYLO', 'STYLO', 'NRC', 'STYLE', 'QUOTE', 'MEN', 'REF']):
            for vf in v.get_feature_names():
                feature_names.append('_'.join([vname, vf]))
        
        return feature_names


