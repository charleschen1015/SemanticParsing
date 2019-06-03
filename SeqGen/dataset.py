# ======================================================================================
# 2019
# Project: Context-Dependent Semantic Parsing
# Author: Charles Chen
# Email: lc971015@ohio.edu
# Paper: Context-Dependent Semantic Parsing over Temporally Structured Data, NAACL 2019.
# ======================================================================================


import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import pdb

from vocabulary import Vocabulary


class DataSet(object):
    def __init__(self,
                 word_idxs1,
                 masks1,
                 len1,
                 batch_size,
                 word_idxs2,
                 masks2,
                 len2,
                 is_train=False,
                 shuffle=False):
        assert(len(word_idxs1) == len(word_idxs2))
        def _build(arr1, arr2):
            co1 = arr1[:-1]
            co2 = arr2[:-1]
            co3 = arr1[1:]
            co4 = arr2[1:]
            return (co1, co2, co3), co4
        self.src, self.dst = _build(word_idxs1, word_idxs2)
        self._src, self._dst = _build(masks1, masks2)
        self._src_, self._dst_ = _build(len1, len2)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()


    def setup(self):
        self.count = len(self.src[0])
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, \
                         self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + \
                           list(np.random.choice(self.count, self.fake_count))
        
        def _tupidx(tup, idx):
            co1, co2, co3 = tup
            return (co1[idx], co2[idx], co3[idx])
            
        src = _tupidx(self.src, current_idxs)
        _src = _tupidx(self._src, current_idxs) #self._src[current_idxs]
        _src_ = _tupidx(self._src_, current_idxs) #self._src_[current_idxs]
        if True: #self.is_train:
            dst = self.dst[current_idxs]
            _dst = self._dst[current_idxs]
            _dst_ = self._dst_[current_idxs]
            self.current_idx += self.batch_size
            return src, _src, _src_, dst, _dst, _dst_
        #else:
        #    self.current_idx += self.batch_size
        #    return src, _src, _src_
        
    def has_next_batch(self):
        return self.current_idx < self.count

    def has_full_next_batch(self):
        return self.current_idx + self.batch_size <= self.count       

    
def read_tex(filename):
    with open(filename, 'r') as file1:
        lines = file1.readlines()
        ct1, ct2 = 0, 0
        src, dst = [], []
        logic, valid = False, False
        tmp = ""
        for line in lines:
            if "textcolor{blue}{" in line:
                line = line.strip()
                s1 = line[line.find("}{") + 2:-1].strip()
                if len(s1) > 0:
                    src.append(s1)
                    ct1 += 1
                    valid = True
            elif "begin{multline*}" in line:
                logic = True
            elif "end{multline*}" in line and logic and valid:
                tmp = tmp.replace("\\", "").replace("==>", " inf ").replace("==", " == ").replace("!=", " neq ").replace("!", " Not ").replace("d.", "e.").replace("d,", "e,").replace("d_1","e_1").replace("(d)", "(e)").replace("e(-1)", " cre ").replace(".", " ldot ").replace("CurrentDate - 1", "CurrentDate-1").replace("CurrentDate + 1", "CurrentDate+1")
                dst.append(tmp.strip())
                logic = False
                valid = False
                tmp = ""
                ct2 += 1
            elif (logic and valid):
                tmp += line.strip()
        print('len', ct1, ct2)
        return src, dst
                
#     


def read_tex_v2(filename):
    with open(filename, 'r') as file1:
        lines = file1.readlines()
        ct1, ct2 = 0, 0
        src, dst = [], []
        logic, valid = False, False
        tmp = ""
        stype = -1
        for line in lines:
            if "%" in line:
                continue
            if "textbf{Query\\theCQuery}" in line or "textbf{Q\\theCQ}" in line:
                stype = 0
            elif "textbf{Statement\\theCStatement}" in line or "textbf{S\\theCS}" in line:
                stype = 1
            elif "textbf{Click\\theCClick}" in line:
                stype = 2
                s1 = ""
                src.append(s1)
                ct1 += 1
                valid = True
            #    
            elif "textcolor{blue}{" in line:
                line = line.strip()
                s1 = line[line.find("}{") + 2:-1].strip()
                if len(s1) > 0:
                    src.append(s1)
                    ct1 += 1
                    valid = True
            elif "begin{multline*}" in line:
                logic = True
            elif "end{multline*}" in line and logic and valid:
                tmp = tmp.replace("\\", "").replace("\mbox{", "").replace("}", "").replace("==", " == ").replace("!=", " neq ").replace("d.", "e.").replace("d,", "e,").replace("d_1","e_1").replace("d_2","e_2").replace("(d)", "(e)").replace("d(-1)", "e(-1)").replace(" d)", " e)").replace("(d ", "(e ").replace("e(-1)", " cre ").replace(".", " ldot ").replace("CurrentDate - 1", "CurrentDate-1").replace("CurrentDate + 1", "CurrentDate+1").replace("mbox{", "")
                dst.append(tmp.strip())
                if stype == 2:
                    src[-1]=tmp.strip()
                    #print(src[-1], dst[-1])
                logic = False
                valid = False
                tmp = ""
                ct2 += 1
            elif (logic and valid):
                tmp += line.strip()
        print('len', ct1, ct2)
        return src, dst
    

        
def prepare_data(config):

    print('Loading data for ' + config.phase)
    if config.phase == 'train':
        filetemp = os.path.join(config.train_dir, config.temp_train_file)
    elif config.phase == 'eval':
        filetemp = os.path.join(config.eval_dir, config.temp_eval_file)
    elif config.phase == 'test':
        filetemp = os.path.join(config.test_dir, config.temp_test_file)
      
    data = np.load(filetemp).item()
    src = data['src']
    dst = data['dst'] 
    
    #
    print("Building the vocabulary...")
    
    vocabulary1 = Vocabulary(config.vocab1_size, save_file = config.vocab1_file)
    #vocabulary1.save(config.vocab1_file)
    print("Vocabulary built.")

    
    #
    if config.phase == 'train':
        filetemp = os.path.join(config.train_dir, config.train_file)
    elif config.phase == 'eval':
        filetemp = os.path.join(config.eval_dir, config.eval_file)
    elif config.phase == 'test':
        filetemp = os.path.join(config.test_dir, config.test_file)
        
    if True: #not os.path.exists(filetemp):
        word_idxs1, word_idxs2 = [], []
        masks1, masks2 = [], []
        len1, len2 = [], []
        for sent in src: #tqdm(src):
            current_word_idxs_ = vocabulary1.process_sentence(sent)
            current_num_words = len(current_word_idxs_)
            #
            len1.append(len(current_word_idxs_))
            #print('len(current_word_idxs_)', len(current_word_idxs_))

            current_word_idxs = np.zeros(config.max_input_length,
                                         dtype = np.int32)
            current_masks = np.zeros(config.max_input_length)
            current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
            current_masks[:current_num_words] = 1.0
            word_idxs1.append(current_word_idxs)
            masks1.append(current_masks)
    
        print('src max length', max(len1))
        #
        #import pdb;pdb.set_trace()
        for sent in dst: #tqdm(dst):
            current_word_idxs_ = vocabulary1.process_sentence(sent + ' stop')
            current_num_words = len(current_word_idxs_)
            #
            len2.append(len(current_word_idxs_))
            #print('len(current_word_idxs_)', len(current_word_idxs_))

            current_word_idxs = np.zeros(config.max_output_length,
                                         dtype = np.int32)
            current_masks = np.zeros(config.max_output_length)
            current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
            current_masks[:current_num_words] = 1.0
            word_idxs2.append(current_word_idxs)
            masks2.append(current_masks)
            
        print('dst max length', max(len2))
        #
        word_idxs1 = np.array(word_idxs1)
        masks1 = np.array(masks1)
        word_idxs2 = np.array(word_idxs2)
        masks2 = np.array(masks2)
        len1 = np.array(len1)
        len2 = np.array(len2)
        data = {'word_idxs1': word_idxs1, 'masks1': masks1, 
                'word_idxs2': word_idxs2, 'masks2': masks2, 
                'len1': len1, 'len2': len2}
        np.save(filetemp, data)
    else:
        data = np.load(filetemp).item()
        word_idxs1 = data['word_idxs1']
        masks1 = data['masks1']
        len1 = data['len1']
        word_idxs2 = data['word_idxs2']
        masks2 = data['masks2']
        len2 = data['len2']
    #    
    print("Building the dataset...")
    is_train = config.phase == 'train'
    dataset = DataSet(word_idxs1, masks1, len1, config.batch_size, word_idxs2, masks2, len2, is_train = is_train, shuffle = is_train)
    print("Dataset built.")
    print("prepare data for " + config.phase + " done!")
    return dataset, vocabulary1#, vocabulary2


       
