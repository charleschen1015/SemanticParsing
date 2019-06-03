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
import tensorflow as tf
import pickle
import copy
import json
from tqdm import tqdm

from nn import NN
from misc import LogicData, TopN


class BaseModel(object):
    def __init__(self, config, vocabulary):
        self.config = config
        self.vocabulary = vocabulary
        self.is_train = True if config.phase == 'train' else False
        self.nn = NN(config)
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.encode_state1, self.encode_state2 = None, None
        self.build()


    def build(self):
        raise NotImplementedError()


    def train(self, sess, train_data):
        raise NotImplementedError()


    def eval(self, sess, ref_data, eval_data, vocabulary):
        raise NotImplementedError()


    def test(self, sess, test_data, vocabulary):
        raise NotImplementedError()

        
    def save(self, sess, saver):
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path + ".npy")))
        #np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        #
        saver.save(sess, save_path + ".ckpt")
        print("Model saved.")
        
        
    def load(self, sess, saver, model_file=None):
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        saver.restore(sess, save_path)
        

    def beam_search(self, sess, batch, vocabulary):            
        config = self.config
        
        (a, b, c), (m_a, m_b, m_c), (l_a_, l_b_, l_c_), dst, m_dst, l_dst_ = batch
        #print(_, a[0], b[0], c[0], m_a[0], m_b[0], m_c[0],  l_a_[0], l_b_[0], l_c_[0], dst[0], m_dst[0], l_dst_[0])
        feed_dict = {self.sentences3: c, self.sequence_length3: l_c_}
                                                        
        initial_memory, initial_output = sess.run(
            [self.initial_memory, self.initial_output],
            feed_dict = feed_dict)
        


        partial_caption_data = []
        complete_caption_data = []
        for k in range(config.batch_size):
            initial_beam = LogicData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

            
        for idx in range(config.max_output_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract() # extract top N * N 
                partial_caption_data_lists.append(data) # len(partial_caption_data_lists): batch_size
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32) # len(last_word): batch_size

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32) # batch_size
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32) # batch_size

                # scores: batch_size * vocab2_size; scores[k]: vocab2_size
                # scores3: batch_size * max_input_length; scores3[k]: max_input_length
                memory, output, scores = sess.run(
                    [self.memory, self.output, self.probs],
                    feed_dict = {self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output})


                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]                    
                    words_and_scores = list(enumerate(scores[k])) # scores: (i.e.prob); words_and_scores:(idx, prob)
              
                        
                    words_and_scores.sort(key=lambda x: -x[1]) # x[1]: prob; x[0]:idx
                    words_and_scores = words_and_scores[0:config.beam_size+1]

                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = LogicData(sentence,
                                           memory[k], # new memory
                                           output[k], # new output
                                           score)
                        #if w >= config.vocab2_size:
                        #    print(w, s)
                        if vocabulary.words[w] == 'stop': # mark the end
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results 


