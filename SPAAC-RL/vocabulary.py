# ======================================================================================
# 2019
# Project: Context-Dependent Semantic Parsing
# Author: Charles Chen
# Email: lc971015@ohio.edu
# Paper: Context-Dependent Semantic Parsing over Temporally Structured Data, NAACL 2019.
# ======================================================================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
from nltk.tokenize import word_tokenize

class Vocabulary(object):
    def __init__(self, size, save_file=None, startIdx = 0):
        self.words = []
        self.word2idx = {}
        self.idx2word = {}
        self.word_frequencies = []
        self.size = size
        if save_file is not None:
            self.load(save_file)
        self.startIdx = startIdx

    def build(self, sentences):
        """ Build the vocabulary and compute the frequency of each word. """
        word_counts = {}
        import pdb;pdb.set_trace()
        for sentence in tqdm(sentences):
            for w in word_tokenize(sentence.lower()):
                if w == '\\\\' or w == ':' or w == 'l' or w == '{' or w == '}':
                    print sentence
                word_counts[w] = word_counts.get(w, 0) + 1.0
        print('len(word_counts.keys())', len(word_counts.keys()))
        assert self.size-1 <= len(word_counts.keys())
        self.words.append('<start>')
        self.word2idx['<start>'] = 0
        self.word_frequencies.append(1.0)

        word_counts = sorted(list(word_counts.items()),
                            key=lambda x: x[1],
                            reverse=True)
        print(word_counts)
        for idx in range(self.size-1):
            word, frequency = word_counts[idx]
            self.words.append(word)
            self.word2idx[word] = idx + 1 + self.startIdx
            self.word_frequencies.append(frequency)

        self.word_frequencies = np.array(self.word_frequencies)
        self.word_frequencies /= np.sum(self.word_frequencies)
        self.word_frequencies = np.log(self.word_frequencies)
        self.word_frequencies -= np.max(self.word_frequencies)

    def process_sentence(self, sentence):
        """ Tokenize a sentence, and translate each token into its index
            in the vocabulary. """
        words = word_tokenize(sentence.lower())
        #print('process_sentence, word_tokenize len(words)', len(words), words)
        word_idxs = [self.word2idx[w] for w in words]
        return word_idxs

    def get_sentence(self, idxs):
        """ Translate a vector of indicies into a sentence. """
        words = [self.idx2word[i] for i in idxs]
        if words[-1] != 'stop':
            words.append('stop')
        length = np.argmax(np.array(words)=='stop') + 1
        words = words[:length]
        sentence = "".join([" "+w if not w.startswith("'") \
                            and w not in string.punctuation \
                            else w for w in words]).strip()
        return sentence

    
    def get_sentence_batch(self, idxs_batch):
        rst = []
        for instance in idxs_batch:
            rst.append(self.get_sentence(instance))
        return rst
    
            
    def save(self, save_file):
        """ Save the vocabulary to a file. """
        data = pd.DataFrame({'word': self.words,
                             'index': list(range(self.startIdx, self.startIdx + self.size)) })
                             #'frequency': self.word_frequencies})
        data.to_csv(save_file)

    def load(self, save_file):
        """ Load the vocabulary from a file. """
        assert os.path.exists(save_file)
        data = pd.read_csv(save_file)
        self.words = data['word'].values
        print('loaded voc size', self.size)
        assert self.size == len(self.words)
        #self.size = len(self.words)
        self.index = data['index'].values # range(self.size) # 
        self.word2idx = dict(zip(self.words, self.index)) #{self.words[i]:i + self.startIdx for i in range(self.size)}
        self.idx2word = dict(zip(self.index, self.words))
        #self.word_frequencies = data['frequency'].values



