# ======================================================================================
# 2019
# Project: Context-Dependent Semantic Parsing
# Author: Charles Chen
# Email: lc971015@ohio.edu
# Paper: Context-Dependent Semantic Parsing over Temporally Structured Data, NAACL 2019.
# ======================================================================================


import tensorflow as tf
import numpy as np

from base_model import *



class Model(BaseModel):
    def build(self):
        self.build_encoder()
        self.build_rnn3()
        if self.config.is_sc:
            self.build_rnn_rl()            
        if self.is_train:
            self.optimizer()
            self.summary()


    def initialize(self, encode_state):
        config = self.config
        if  encode_state is None:
            memory = tf.zeros([config.batch_size, config.num_lstm_units])
            output = tf.zeros([config.batch_size, config.num_lstm_units])
        else:
            encoding1 = self.nn.dropout(encode_state[0])
            encoding2 = self.nn.dropout(encode_state[1])
            memory = self.nn.dense(encoding1,
                                       units = config.num_lstm_units,
                                       activation = None,
                                       name = 'fc_a')
            output = self.nn.dense(encoding2,
                                       units = config.num_lstm_units,
                                       activation = None,
                                       name = 'fc_b')
            #memory = tf.tile(memory, [config.batch_size, 1])
            #output = tf.tile(output, [config.batch_size, 1])
        return tf.contrib.rnn.LSTMStateTuple(memory, output)
       

    def build_rnn1(self, sentences, sequence_length, encode_state):
        config = self.config

        with tf.variable_scope("word_embedding", reuse = tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocab1_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)
            word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                       sentences)
                
        lstm = tf.contrib.rnn.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer, state_is_tuple = True)

        if self.is_train:
            lstm = tf.contrib.rnn.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0 - config.lstm_drop_rate,
                output_keep_prob = 1.0 - config.lstm_drop_rate,
                state_keep_prob = 1.0 - config.lstm_drop_rate)
            print('rnn1 use dropout')


        with tf.variable_scope("initialize1", reuse = tf.AUTO_REUSE):
            initial_state = self.initialize(encode_state)
            print("initial_state", initial_state)
                
        # both tuple: outputs, output_states
        with tf.variable_scope("rnn1", reuse = tf.AUTO_REUSE):
            outputs, output_states  = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = lstm, cell_bw = lstm,
            initial_state_fw = initial_state, initial_state_bw = initial_state,
            dtype = tf.float32,
            inputs = word_embed,
            sequence_length = sequence_length, time_major = False)

        contexts = tf.concat(outputs, 2)
        print("contexts", contexts)

        print("output_states", output_states)
        output_state_fw, output_state_bw = output_states
        t1 = tf.concat((output_state_fw[0], output_state_bw[0]),  1)
        t2 = tf.concat((output_state_fw[1], output_state_bw[1]),  1)
        new_encode_state = t1, t2
        print("encode_state", new_encode_state)
               
        print("RNN1 built.")
        return contexts, new_encode_state


    
    def build_rnn2(self, sentences2, sequence_length2, encode_state2):
        config = self.config

        with tf.variable_scope("word_embedding", reuse = tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocab1_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)
            word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    sentences2)
                
        lstm = tf.contrib.rnn.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer, state_is_tuple = True)

        if self.is_train:
            lstm = tf.contrib.rnn.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0 - config.lstm_drop_rate,
                output_keep_prob = 1.0 - config.lstm_drop_rate,
                state_keep_prob = 1.0 - config.lstm_drop_rate)
            print('rnn2 use dropout')


        with tf.variable_scope("initialize2"):
            initial_state = self.initialize(encode_state2)
            print("initial_state", initial_state)
                
        # both tuple: outputs, output_states
        with tf.variable_scope("rnn2"):
            outputs, output_states  = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = lstm, cell_bw = lstm,
            initial_state_fw = initial_state, initial_state_bw = initial_state,
            dtype = tf.float32,
            inputs = word_embed,
            sequence_length = sequence_length2, time_major = False)

        contexts2 = tf.concat(outputs, 2) 
        print("contexts2", contexts2)

        print("output_states", output_states)
        output_state_fw, output_state_bw = output_states
        t1 = tf.concat((output_state_fw[0], output_state_bw[0]),  1)
        t2 = tf.concat((output_state_fw[1], output_state_bw[1]),  1)
        new_encode_state2 = t1, t2
        print("encode_state", new_encode_state2)
        print("RNN2 built.")
        return contexts2, new_encode_state2
        
    
    def build_encoder(self):
        config = self.config
        self.sentences1 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size, config.max_input_length])
        self.sequence_length1 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size])

        self.sentences2 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size, config.max_output_length])
        self.sequence_length2 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size])

        self.sentences3 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size, config.max_input_length])
        self.sequence_length3 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size])
        
        self.contexts1, self.encode_state1 = self.build_rnn1(self.sentences1, self.sequence_length1, self.encode_state1)
        self.contexts2, self.encode_state2 = self.build_rnn2(self.sentences2, self.sequence_length2, self.encode_state1)
        self.contexts3, self.encode_state1 = self.build_rnn1(self.sentences3, self.sequence_length3, self.encode_state1)
        print('Encoder built..')
        
        
    def attend(self, contexts, output, dim_ctx, num_ctx):
        config = self.config
        reshaped_contexts = tf.reshape(contexts, [-1, dim_ctx])# all attentions in one batch [batch_size * num_ctx, dim_ctx]
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)
        if config.num_attend_layers == 2:
            temp1 = self.nn.dense(reshaped_contexts, 
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1a') 
            #
            temp2 = self.nn.dense(output,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1b') 
            temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, num_ctx, 1]) 
            temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer]) 
            temp = temp1 + temp2 
            temp = self.nn.dropout(temp) # [batch_size * num_ctx, dim_attend_layer]
            logits = self.nn.dense(temp,
                                   units = 1,
                                   activation = None,
                                   use_bias = False,
                                   name = 'fc_2')
            logits = tf.reshape(logits, [-1, num_ctx])
        alpha = tf.nn.softmax(logits) # [batch_size, num_ctx]
        return alpha
    
    
    def decode(self, expanded_output):
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 2:
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocab1_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits
    
    
    def decode_v2(self, output, context1, context2, context3, cont1, cont2, cont3):
        config = self.config
        v1 = tf.concat([output, context1, context2, context3], axis = 1) # generate v1: [batch_size, long]
        v2 = tf.concat([output, context2], axis = 1) # coref v2: [batch_size, long]
        v3 = tf.concat([output, context3], axis = 1) # copy v3: [batch_size, long]
        # generate
        logits1 = self.nn.dense(v1,
                                units = config.vocab2_size, 
                                activation = None,
                                name = 'generate')

        # coref
        # cont2: [batch_size, max_output_length, dim]
        num_ctx = config.max_output_length
        dim_ctx = config.num_lstm_units * 2
        reshaped_contexts = tf.reshape(cont2, [-1, dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        
        temp1 = self.nn.dense(reshaped_contexts, 
                              units = config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'coref_1a') 
        
        v2 = self.nn.dropout(v2)
        temp2 = self.nn.dense(v2,
                              units = config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'coref_1b') 
        temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, num_ctx, 1]) 
        temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer]) 
        temp = temp1 + temp2 
        temp = self.nn.dropout(temp) 
        logits2 = self.nn.dense(temp,
                                units = 1,
                                activation = None,
                                use_bias = False,
                                name = 'coref_2') 
        logits2 = tf.reshape(logits2, [-1, num_ctx]) # [batch_size, num_ctx], i.e. [batch_size, max_output_length]       
              
        # copy
        # cont3: [batch_size, max_input_length, dim]
        num_ctx = config.max_input_length
        dim_ctx = config.num_lstm_units * 2
        reshaped_contexts = tf.reshape(cont3, [-1, dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        
        temp1 = self.nn.dense(reshaped_contexts, 
                              units = config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'copy_1a') 
        
        v3 = self.nn.dropout(v3)
        temp2 = self.nn.dense(v3,
                              units = config.dim_attend_layer,
                              activation = tf.tanh,
                              name = 'copy_1b') 
        temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, num_ctx, 1]) 
        temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer]) 
        temp = temp1 + temp2 
        temp = self.nn.dropout(temp) 
        logits3 = self.nn.dense(temp,
                                units = 1,
                                activation = None,
                                use_bias = False,
                                name = 'copy_2') 
        logits3 = tf.reshape(logits3, [-1, num_ctx]) # [batch_size, num_ctx], i.e. [batch_size, max_input_length]

        #
        return logits1, logits2, logits3

    
        
    def get_l(self, sent4, sentences2, sentences3):
        # sent4: [batch_size]
        # sentences2: [batch_size, max_output_length] 
        # sentences3: [batch_size, max_input_length] 
        config = self.config
        l = []
        for idx, w in enumerate(sent4):
            l1 = tf.zeros([config.vocab2_size], tf.float32) #  num of tokens in logical forms
            l2 = tf.zeros([config.max_output_length], tf.float32) #  coref
            l3 = tf.zeros([config.max_input_length], tf.float32) #  copy
            if w < 52:
                l1[w] = 1.
            elif w == 52:
                l2[tf.argmax(tf.cast(tf.equal(sentences2[idx], 8), tf.int32), axis = 0)] = 1.
            else:
                l3[tf.argmax(tf.cast(tf.equal(sentences3[idx], w), tf.int32), axis = 0)] = 1.
            l.append(tf.concat([l1, l2, l3]))
        l = tf.stack(l, axis = 0) # [batch_size, vocab2_size + max_output_length + max_input_length]
        l = tf.argmax(l, axis = 1)
        return l
    
    
    
    def get_l_v2(self, sent4, sentences2, sentences3):
        config = self.config
        
        
        masked = tf.less(sent4, config.vocab2_size)
        oov = tf.zeros_like(sent4) + config.vocab2_size - 1
        t1 = tf.where(masked, sent4, oov)
        t1 = tf.one_hot(t1, config.vocab2_size) # [batch_size, vocab2_size] 
        l1 = t1
        
        t1 = tf.cast(tf.equal(sentences2, self.vocabulary.word2idx['e']), tf.float32) 
        t2 = tf.tile(tf.expand_dims(tf.cast(tf.equal(sent4, self.vocabulary.word2idx['copy'] - 1), tf.float32), 1), [1, config.max_output_length]) ## !!! index for word 'cre'
        l2 = t1 * t2 
        #
        t0 = tf.tile(tf.expand_dims(tf.cast(sent4, tf.float32), 1), [1, config.max_input_length])
        t1 = tf.cast(tf.equal(tf.cast(sentences3, tf.float32), t0), tf.float32) # [batch_size, max_input_length]
        t2 = tf.tile(tf.expand_dims(tf.cast(tf.greater(sent4, self.vocabulary.word2idx['copy']), tf.float32), 1), [1, config.max_input_length]) ## !!! index for word 'copy'
        #
        l3 = t1 * t2  
        #print(l)
        l1 = tf.argmax(l1, axis = 1)
        return l1, l2, l3
    
    
    
    def build_rnn3(self):
        config = self.config

        if self.is_train and not config.is_sc:
            cont1, cont2, cont3 = self.contexts1, self.contexts2, self.contexts3
            #
            sentences4 = tf.placeholder(dtype = tf.int32, shape = [None, config.max_output_length])
            #
            masks4 = tf.placeholder(dtype = tf.float32, shape = [None, config.max_output_length])
            #
            masks2 = tf.placeholder(dtype = tf.float32, shape = [None, config.max_output_length])
            masks3 = tf.placeholder(dtype = tf.float32, shape = [None, config.max_input_length])
        else:
            dim_ctx = config.num_lstm_units * 2

                
            cont1 = tf.placeholder(dtype = tf.float32, 
                                   shape = [None, config.max_input_length, dim_ctx])
            cont2 = tf.placeholder(dtype = tf.float32, 
                                   shape = [None, config.max_output_length, dim_ctx])
            cont3 = tf.placeholder(dtype = tf.float32, 
                                   shape = [None, config.max_input_length, dim_ctx])
                    
            last_memory = tf.placeholder(dtype = tf.float32,
                                         shape = [None, config.num_lstm_units * 2])
            last_output = tf.placeholder(dtype = tf.float32,
                                         shape = [None, config.num_lstm_units * 2])
            last_word = tf.placeholder(dtype = tf.int64,
                                         shape = [None])

        with tf.variable_scope("word_embedding", reuse = tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocab1_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train and not config.is_sc)
                
        lstm = tf.contrib.rnn.LSTMCell(
            config.num_lstm_units * 2,
            initializer = self.nn.fc_kernel_initializer, state_is_tuple = True)

        if self.is_train and not config.is_sc:
            lstm = tf.contrib.rnn.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0 - config.lstm_drop_rate,
                output_keep_prob = 1.0 - config.lstm_drop_rate,
                state_keep_prob = 1.0 - config.lstm_drop_rate)
            print('rnn3 use dropout')


        with tf.variable_scope("initialize3_1"):
            initial_state1 = self.initialize(self.encode_state1)
            initial_memory1, initial_output1 = initial_state1
            
        with tf.variable_scope("initialize3_2"):
            initial_state2 = self.initialize(self.encode_state2)
            initial_memory2, initial_output2 = initial_state2
            initial_memory = tf.concat([initial_memory1, initial_memory2], 1)
            initial_output = tf.concat([initial_output1, initial_output2], 1)
            initial_state = tf.contrib.rnn.LSTMStateTuple(initial_memory, initial_output)
            print("initial_state", initial_state)        
        
        
        predictions = []
        if self.is_train and not config.is_sc:
            alphas1, alphas2, alphas3 = [], [], []
            _labels, cross_entropies, entropies2, entropies3 = [], [], [], []
            predictions_correct = []
            num_steps = config.max_output_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
        last_state = last_memory, last_output
        
        
                
        for idx in range(num_steps):

            with tf.variable_scope("attend1"):
                num_ctx = config.max_input_length                
                alpha1 = self.attend(cont1, last_output, config.dim_attend_layer, num_ctx)
                #
                context1 = tf.reduce_sum(cont1 * tf.expand_dims(alpha1, 2),
                                        axis = 1)
                if self.is_train and not config.is_sc:
                    tiled_masks = tf.tile(tf.expand_dims(masks4[:, idx], 1),
                                         [1, num_ctx]) # [batch_size, num_ctx]
                    masked_alpha1 = alpha1 * tiled_masks # [batch_size, num_ctx]
                    alphas1.append(tf.reshape(masked_alpha1, [-1]))
                    
                  
            with tf.variable_scope("attend2"):
                num_ctx = config.max_output_length                
                alpha2 = self.attend(cont2, last_output, config.dim_attend_layer, num_ctx)
                context2 = tf.reduce_sum(cont2 * tf.expand_dims(alpha2, 2),
                                        axis = 1)
                if self.is_train and not config.is_sc:
                    tiled_masks = tf.tile(tf.expand_dims(masks4[:, idx], 1),
                                         [1, num_ctx]) # [batch_size, num_ctx]
                    masked_alpha2 = alpha2 * tiled_masks # [batch_size, num_ctx]
                    alphas2.append(tf.reshape(masked_alpha2, [-1]))
                    
                    
            with tf.variable_scope("attend3"):
                num_ctx = config.max_input_length                
                alpha3 = self.attend(cont3, last_output, config.dim_attend_layer, num_ctx)
                context3 = tf.reduce_sum(cont3 * tf.expand_dims(alpha3, 2),
                                        axis = 1)              
                if self.is_train and not config.is_sc:
                    tiled_masks = tf.tile(tf.expand_dims(masks4[:, idx], 1),
                                         [1, num_ctx]) # [batch_size, num_ctx]
                    masked_alpha3 = alpha3 * tiled_masks # [batch_size, num_ctx]
                    alphas3.append(tf.reshape(masked_alpha3, [-1]))
                                       
            #
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)

            with tf.variable_scope("rnn3"):
                current_input = tf.concat([context1, context2, context3, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            
            with tf.variable_scope("decode"):
                #                
                logits1, logits2, logits3 = self.decode_v2(output, context1, context2, context3, cont1, cont2, cont3)
                #decode(expanded_output)
                probs = tf.nn.softmax(logits1) #logits  batch_size * vocab2_size
                probs2 = tf.nn.softmax(logits2) # coref batch_size * max_output_length
                probs3 = tf.nn.softmax(logits3) # copy  batch_size * max_input_length
                #
                _prediction = tf.argmax(logits1, 1) #_prediction batch_size
                _masks = tf.less(_prediction, config.vocab2_size - 1)
                #
                _prediction3 = tf.argmax(logits3, 1) # _prediction3 batch_size
                _masks3 = tf.one_hot(_prediction3, config.max_input_length) # batch_size max_input_length
                w_c = tf.reduce_max(tf.cast(self.sentences3, tf.float32) * _masks3, 1) # batch_size
                
                prediction = tf.where(_masks, _prediction, tf.cast(w_c, tf.int64)) 
                                
                #
                predictions.append(prediction)

                #
                _sample_word = tf.multinomial(logits1, config.beam_size, name = 'multinomial')
                _sample_masks = tf.less(_sample_word, config.vocab2_size - 1)
                sample_word = _sample_word#tf.where(_sample_masks, _sample_word, tf.cast(tf.tile(tf.expand_dims(w_c, -1), [1, config.beam_size]), tf.int64)) 
                  
                
            if self.is_train and not config.is_sc:
                labels1, labels2, labels3 = self.get_l_v2(sentences4[:, idx], self.sentences2, self.sentences3) # [batch_size, long]
                _labels.append(labels1)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = labels1, #sentences4[:, idx],
                    logits = logits1)
                masked_cross_entropy = cross_entropy * masks4[:, idx]
                cross_entropies.append(masked_cross_entropy)
               
                entropy2 = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = labels2,
                    logits = logits2)
                masked_entropy2 = tf.reduce_sum(entropy2 * masks2, axis = 1) / tf.reduce_sum(masks2, axis = 1)
                masked_entropy2 = masked_entropy2 * masks4[:, idx]
                entropies2.append(masked_entropy2)
                
                entropy3 = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = labels3,
                    logits = logits3)
                masked_entropy3 = tf.reduce_sum(entropy3 * masks3, axis = 1) / tf.reduce_sum(masks3, axis = 1)
                masked_entropy3 = masked_entropy3 * masks4[:, idx]
                entropies3.append(masked_entropy3)


            
                ground_truth = tf.cast(sentences4[:, idx], tf.int64) # labels1
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(masks4[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                last_output = output
                last_memory = memory
                last_state = state
                last_word = sentences4[:, idx]
            tf.get_variable_scope().reuse_variables()
            
        #    
        if self.is_train and not config.is_sc:
            self._labels = tf.stack(_labels, axis = 1)
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks4)
                
            entropies2 = tf.stack(entropies2, axis = 1)
            entropy_loss2 = tf.reduce_sum(entropies2) \
                                 / tf.reduce_sum(masks4)
                               
            entropies3 = tf.stack(entropies3, axis = 1)
            entropy_loss3 = tf.reduce_sum(entropies3) \
                                 / tf.reduce_sum(masks4)
                
                
            #
            num_ctx = config.max_input_length
            alphas1 = tf.stack(alphas1, axis = 1)
            # alphas1: max_caption_length [batch_size, num_ctx] 
            alphas1 = tf.reshape(alphas1, [config.batch_size, num_ctx, -1]) 
            # switch the last two dims ==> [batch_size, num_ctx, max_caption_length]
            attentions1 = tf.reduce_sum(alphas1, axis = 2) # [batch_size, num_ctx] 
            diffs1 = tf.ones_like(attentions1) - attentions1
            attention_loss1 = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs1) \
                             / (config.batch_size * num_ctx)
            #
            num_ctx = config.max_output_length
            alphas2 = tf.stack(alphas2, axis = 1) 
            # alphas2: max_caption_length [batch_size, num_ctx] 
            alphas2 = tf.reshape(alphas2, [config.batch_size, num_ctx, -1]) 
            # switch the last two dims ==> [batch_size, num_ctx, max_caption_length]
            attentions2 = tf.reduce_sum(alphas2, axis = 2) # [batch_size, num_ctx] 
            diffs2 = tf.ones_like(attentions2) - attentions2 
            attention_loss2 = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs2) \
                             / (config.batch_size * num_ctx)            
            #
            num_ctx = config.max_input_length
            alphas3 = tf.stack(alphas3, axis = 1) 
            # alphas3: max_caption_length [batch_size, num_ctx] 
            alphas3 = tf.reshape(alphas3, [config.batch_size, num_ctx, -1]) 
            # switch the last two dims ==> [batch_size, num_ctx, max_caption_length]
            attentions3 = tf.reduce_sum(alphas3, axis = 2) # [batch_size, num_ctx] 
            diffs3 = tf.ones_like(attentions3) - attentions3 
            attention_loss3 = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs3) \
                             / (config.batch_size * num_ctx)       
                    
            #
            reg_loss = tf.losses.get_regularization_loss()
            total_loss = cross_entropy_loss +  entropy_loss2 + entropy_loss3 \
                         + reg_loss + attention_loss1  + attention_loss2 + attention_loss3
            
            predictions_correct = tf.stack(predictions_correct, axis = 1) # [batch_size, max_caption_length]
            accuracy = tf.reduce_sum(tf.cast(predictions_correct, tf.float32)) \
                       / tf.reduce_sum(masks4)
        
        self.b_ctx1, self.b_ctx2, self.b_ctx3 = cont1, cont2, cont3
        self.predictions = tf.stack(predictions, axis = 1)
        if self.is_train and not config.is_sc:
            self.sentences4 = sentences4
            self.masks4 = masks4
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
            
            self.masks2 = masks2
            self.masks3 = masks3
            self.entropy_loss2 = entropy_loss2
            self.entropy_loss3 = entropy_loss3
            
            self.attention_loss1 = attention_loss1
            self.attention_loss2 = attention_loss2 
            self.attention_loss3 = attention_loss3            
            self.reg_loss = reg_loss
            self.accuracy = accuracy
            self.perplexity = tf.exp(cross_entropy_loss)
            self.attentions1 = attentions1
            self.attentions2 = attentions2
            self.attentions3 = attentions3 
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output

            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs    
            self.probs2 = probs2   
            self.probs3 = probs3
            #
            self.sample_word = sample_word
        print("RNN3 built.")
            

    def build_rnn_rl(self):
        print("Building the RNN RL...")
        config = self.config

        if config.is_sc:
            #
            cont1, cont2, cont3 = self.contexts1, self.contexts2, self.contexts3
            samples = tf.placeholder(dtype = tf.int32, shape = [None, config.max_output_length])
            maskrl = tf.placeholder(dtype = tf.float32, shape = [None, config.max_output_length])
            #
            masks2 = tf.placeholder(dtype = tf.float32, shape = [None, config.max_output_length])
            masks3 = tf.placeholder(dtype = tf.float32, shape = [None, config.max_input_length])
            #
            rewards = tf.placeholder(dtype = tf.float32, shape = [None], name = 'rewards') # config.batch_size


        with tf.variable_scope("word_embedding", reuse = tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocab1_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = config.is_sc)
                
        lstm = tf.contrib.rnn.LSTMCell(
            config.num_lstm_units * 2,
            initializer = self.nn.fc_kernel_initializer, state_is_tuple = True)

        if config.is_sc:
            lstm = tf.contrib.rnn.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0 - config.lstm_drop_rate,
                output_keep_prob = 1.0 - config.lstm_drop_rate,
                state_keep_prob = 1.0 - config.lstm_drop_rate)
            print('rnn rl use dropout')


        with tf.variable_scope("initialize3_1"):
            initial_state1 = self.initialize(self.encode_state1)
            initial_memory1, initial_output1 = initial_state1
            
        with tf.variable_scope("initialize3_2"):
            initial_state2 = self.initialize(self.encode_state2)
            initial_memory2, initial_output2 = initial_state2
            initial_memory = tf.concat([initial_memory1, initial_memory2], 1)
            initial_output = tf.concat([initial_output1, initial_output2], 1)
            initial_state = tf.contrib.rnn.LSTMStateTuple(initial_memory, initial_output)
            print("rl initial_state", initial_state)        
            
            
        predictions = []
        if config.is_sc:
            alphas1, alphas2, alphas3 = [], [], []
            _labels, cross_entropies, entropies2, entropies3 = [], [], [], []
            predictions_correct = []
            num_steps = config.max_output_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
        last_state = last_memory, last_output           
            
            
        for idx in range(num_steps):

            with tf.variable_scope("attend1"):
                num_ctx = config.max_input_length                
                alpha1 = self.attend(cont1, last_output, config.dim_attend_layer, num_ctx)
                #print('alpha1', alpha1)
                context1 = tf.reduce_sum(cont1 * tf.expand_dims(alpha1, 2),
                                        axis = 1)
                if config.is_sc:
                    tiled_masks = tf.tile(tf.expand_dims(maskrl[:, idx], 1),
                                         [1, num_ctx]) # [batch_size, num_ctx]
                    masked_alpha1 = alpha1 * tiled_masks # [batch_size, num_ctx]
                    alphas1.append(tf.reshape(masked_alpha1, [-1]))
                    
                  
            with tf.variable_scope("attend2"):
                num_ctx = config.max_output_length                
                alpha2 = self.attend(cont2, last_output, config.dim_attend_layer, num_ctx)
                context2 = tf.reduce_sum(cont2 * tf.expand_dims(alpha2, 2),
                                        axis = 1)
                if config.is_sc:
                    tiled_masks = tf.tile(tf.expand_dims(maskrl[:, idx], 1),
                                         [1, num_ctx]) # [batch_size, num_ctx]
                    masked_alpha2 = alpha2 * tiled_masks # [batch_size, num_ctx]
                    alphas2.append(tf.reshape(masked_alpha2, [-1]))
                    
                    
            with tf.variable_scope("attend3"):
                num_ctx = config.max_input_length                
                alpha3 = self.attend(cont3, last_output, config.dim_attend_layer, num_ctx)
                context3 = tf.reduce_sum(cont3 * tf.expand_dims(alpha3, 2),
                                        axis = 1)              
                if config.is_sc:
                    tiled_masks = tf.tile(tf.expand_dims(maskrl[:, idx], 1),
                                         [1, num_ctx]) # [batch_size, num_ctx]
                    masked_alpha3 = alpha3 * tiled_masks # [batch_size, num_ctx]
                    alphas3.append(tf.reshape(masked_alpha3, [-1]))            
            
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)

            with tf.variable_scope("rnn3"):
                current_input = tf.concat([context1, context2, context3, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context1, context2, context3,
                                             word_embed],
                                             axis = 1)
                logits1, logits2, logits3 = self.decode_v2(output, context1, context2, context3, cont1, cont2, cont3) 
                #decode(expanded_output)
                probs = tf.nn.softmax(logits1) #logits  batch_size * vocab2_size
                probs2 = tf.nn.softmax(logits2) # coref batch_size * max_output_length
                probs3 = tf.nn.softmax(logits3) # copy  batch_size * max_input_length
                #
                _prediction = tf.argmax(logits1, 1) #_prediction batch_size
                _masks = tf.less(_prediction, config.vocab2_size - 1)
                #
                _prediction3 = tf.argmax(logits3, 1) # _prediction3 batch_size
                _masks3 = tf.one_hot(_prediction3, config.max_input_length) # batch_size max_input_length
                w_c = tf.reduce_max(tf.cast(self.sentences3, tf.float32) * _masks3, 1) # batch_size
                
                prediction = tf.where(_masks, _prediction, tf.cast(w_c, tf.int64)) 
                                
                #
                predictions.append(prediction)

                
            if config.is_sc:
                labels1, labels2, labels3 = self.get_l_v2(samples[:, idx], self.sentences2, self.sentences3) # [batch_size, long]
                _labels.append(labels1)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = labels1,
                    logits = logits1)
                masked_cross_entropy = cross_entropy * maskrl[:, idx]
                cross_entropies.append(masked_cross_entropy)
               
                entropy2 = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = labels2,
                    logits = logits2)
                masked_entropy2 = tf.reduce_sum(entropy2 * masks2, axis = 1) / tf.reduce_sum(masks2, axis = 1)
                masked_entropy2 = masked_entropy2 * maskrl[:, idx]
                entropies2.append(masked_entropy2)
                
                entropy3 = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = labels3,
                    logits = logits3)
                masked_entropy3 = tf.reduce_sum(entropy3 * masks3, axis = 1) / tf.reduce_sum(masks3, axis = 1)
                masked_entropy3 = masked_entropy3 * maskrl[:, idx]
                entropies3.append(masked_entropy3)


            
                ground_truth = tf.cast(samples[:, idx], tf.int64) # labels1
                prediction_correct = tf.where(
                    tf.equal(prediction, ground_truth),
                    tf.cast(maskrl[:, idx], tf.float32),
                    tf.cast(tf.zeros_like(prediction), tf.float32))
                predictions_correct.append(prediction_correct)

                last_output = output
                last_memory = memory
                last_state = state
                last_word = samples[:, idx]
            tf.get_variable_scope().reuse_variables()
            
            
        if config.is_sc:
            self.rl_labels = tf.stack(_labels, axis = 1)
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(maskrl)
                
            entropies2 = tf.stack(entropies2, axis = 1)
            entropy_loss2 = tf.reduce_sum(entropies2) \
                                 / tf.reduce_sum(maskrl)
                               
            entropies3 = tf.stack(entropies3, axis = 1)
            entropy_loss3 = tf.reduce_sum(entropies3) \
                                 / tf.reduce_sum(maskrl)
                
                
            #
            num_ctx = config.max_input_length
            alphas1 = tf.stack(alphas1, axis = 1)
            # alphas1: max_caption_length [batch_size, num_ctx] 
            alphas1 = tf.reshape(alphas1, [config.batch_size, num_ctx, -1]) 
            # switch the last two dims ==> [batch_size, num_ctx, max_caption_length]
            attentions1 = tf.reduce_sum(alphas1, axis = 2) # [batch_size, num_ctx] 
            diffs1 = tf.ones_like(attentions1) - attentions1
            attention_loss1 = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs1) \
                             / (config.batch_size * num_ctx)
            #
            num_ctx = config.max_output_length
            alphas2 = tf.stack(alphas2, axis = 1) 
            # alphas2: max_caption_length [batch_size, num_ctx] 
            alphas2 = tf.reshape(alphas2, [config.batch_size, num_ctx, -1]) 
            # switch the last two dims ==> [batch_size, num_ctx, max_caption_length]
            attentions2 = tf.reduce_sum(alphas2, axis = 2) # [batch_size, num_ctx] 
            diffs2 = tf.ones_like(attentions2) - attentions2 
            attention_loss2 = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs2) \
                             / (config.batch_size * num_ctx)            
            #
            num_ctx = config.max_input_length
            alphas3 = tf.stack(alphas3, axis = 1) 
            # alphas3: max_caption_length [batch_size, num_ctx] 
            alphas3 = tf.reshape(alphas3, [config.batch_size, num_ctx, -1]) 
            # switch the last two dims ==> [batch_size, num_ctx, max_caption_length]
            attentions3 = tf.reduce_sum(alphas3, axis = 2) # [batch_size, num_ctx] 
            diffs3 = tf.ones_like(attentions3) - attentions3 
            attention_loss3 = config.attention_loss_factor \
                             * tf.nn.l2_loss(diffs3) \
                             / (config.batch_size * num_ctx)       
                    
            #
            reg_loss = tf.losses.get_regularization_loss()
            total_loss = cross_entropy_loss +  entropy_loss2 + entropy_loss3 \
                         + reg_loss + attention_loss1  + attention_loss2 + attention_loss3
            
            predictions_correct = tf.stack(predictions_correct, axis = 1) # [batch_size, max_caption_length]
            accuracy = tf.reduce_sum(tf.cast(predictions_correct, tf.float32)) \
                       / tf.reduce_sum(maskrl)
        
        self.rl_ctx1, self.rl_ctx2, self.rl_ctx3 = cont1, cont2, cont3
        self.rl_predictions = tf.stack(predictions, axis = 1)
        if config.is_sc:           
            self.rewards = rewards
            self.samples = samples
            self.maskrl = maskrl
            #           
            self.rl_loss = total_loss
            self.rl_cross_entropy_loss = cross_entropy_loss
            
            self.rl_masks2 = masks2
            self.rl_masks3 = masks3
            self.rl_entropy_loss2 = entropy_loss2
            self.rl_entropy_loss3 = entropy_loss3
            
            self.rl_attention_loss1 = attention_loss1
            self.rl_attention_loss2 = attention_loss2 
            self.rl_attention_loss3 = attention_loss3            
            self.rl_reg_loss = reg_loss
            self.rl_accuracy = accuracy
            self.rl_perplexity = tf.exp(cross_entropy_loss)
            self.rl_attentions1 = attentions1
            self.rl_attentions2 = attentions2
            self.rl_attentions3 = attentions3         
        print("RNN3 RL built.")
            
            
            
    def optimizer(self):
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        rl_learning_rate = tf.constant(config.rl_initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
                
            if self.is_train and not self.config.is_sc:
                opt_op = tf.contrib.layers.optimize_loss(
                    loss = self.total_loss,
                    global_step = self.global_step,
                    learning_rate = learning_rate,
                    optimizer = optimizer,
                    clip_gradients = config.clip_gradients,
                    learning_rate_decay_fn = learning_rate_decay_fn)
                self.opt_op = opt_op                
                
            if self.config.is_sc:
                opt_sc = tf.contrib.layers.optimize_loss(
                    loss = self.rl_loss,
                    global_step = self.global_step,
                    learning_rate = rl_learning_rate,
                    optimizer = optimizer,
                    clip_gradients = config.clip_gradients,
                    learning_rate_decay_fn = learning_rate_decay_fn)
                self.opt_sc = opt_sc
        print('Optimizor built..')

                 
            
    def summary(self):
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)
                    
        if self.is_train and not self.config.is_sc:
            with tf.name_scope("metrics"):
                tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)


                tf.summary.scalar("entropy_loss2", self.entropy_loss2)
                tf.summary.scalar("entropy_loss3", self.entropy_loss3)


                tf.summary.scalar("attention_loss1", self.attention_loss1)
                tf.summary.scalar("attention_loss2", self.attention_loss2)
                tf.summary.scalar("attention_loss3", self.attention_loss3)
                tf.summary.scalar("reg_loss", self.reg_loss)
                tf.summary.scalar("total_loss", self.total_loss)
                tf.summary.scalar("accuracy", self.accuracy)
        if self.config.is_sc:
            with tf.name_scope("metrics3"):
                tf.summary.scalar("RL_total_loss", self.rl_loss)          
                print("RL summary")
        self.summary = tf.summary.merge_all()
        print('Summary built..')

        
    def variable_summary(self, var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

        
        
    def train(self, sess, saver, train_data, vocabulary):
        print("Training the model...")
        config = self.config
        #
        #
        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)
        #
        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            for _ in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                (a, b, c), (m_a, m_b, m_c), (l_a_, l_b_, l_c_), dst, m_dst, l_dst_ = batch
                feed_dict = {self.sentences1: a, self.sequence_length1: l_a_,
                             self.sentences2: b, self.sequence_length2: l_b_,
                             self.sentences3: c, self.sequence_length3: l_c_,
                             self.sentences4: dst, self.masks4: m_dst, self.masks2: m_b, self.masks3: m_c}
                _, summary, global_step, predic, acc, _labels = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step, self.predictions, self.accuracy, 
                                                                                   self._labels],
                                                    feed_dict = feed_dict)
    
                #
                if (global_step + 1) % config.save_period == 0:
                    
                    self.save(sess, saver)
                    
                train_writer.add_summary(summary, global_step)
            train_data.reset()
          
            #
            #self.eval(sess, ref_cap, eval_data, vocabulary)
            #eval_data.reset()
            #
        self.save(sess, saver)           
        train_writer.close()
        print("Training complete.")


    def train_sc(self, sess, saver, train_data, vocabulary, eval_data):
        def _pad(arr, batch_size):
            rst = []
            n = len(arr)
            for it in arr:
                rst.append(it)
            if n != batch_size:
                for _ in range(batch_size - n):
                    rst.append(arr[-1])
            rst = np.array(rst)
            return rst
        #        
        print("Training the model rl ...")
        config = self.config
        #
        #
        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir,
                                             sess.graph)
        #
        
        for _ in tqdm(list(range(config.num_epochs)), desc='epoch'):
            idx = 0
            for k in tqdm(list(range(train_data.num_batches)), desc='batch'):
                batch = train_data.next_batch()
                (a, b, c), (m_a, m_b, m_c), (l_a_, l_b_, l_c_), dst, m_dst, l_dst_ = batch
                #
                feed_dict = {self.sentences1: a, self.sequence_length1: l_a_,
                             self.sentences2: b, self.sequence_length2: l_b_,
                             self.sentences3: c, self.sequence_length3: l_c_}
                cont1, cont2, cont3, initial_memory, initial_output = sess.run(
                    [self.contexts1, self.contexts2, self.contexts3,  
                                     self.initial_memory, self.initial_output],
                    feed_dict = feed_dict)
                #               
                decode_greedy = self.beam_greedy(sess, cont1, cont2, cont3, initial_memory, initial_output, vocabulary) 
                print('greedy done') 
                decode_sample = self.beam_sample(sess, cont1, cont2, cont3, initial_memory, initial_output, vocabulary) 
                print('sample done') 
                #
                fake_cnt = 0 if k < train_data.num_batches-1 \
                             else train_data.fake_count
                maskrl, sample_feed = [], []
                ref, hypo_greedy, hypo_sample = {}, {}, {}
                accs_g, accs_s = [], []
                for l in range(train_data.batch_size-fake_cnt):
                    greedy_idxs = decode_greedy[l][0].sentence 
                    _greedy = vocabulary.get_sentence(greedy_idxs)
                    sample_idxs = decode_sample[l][0].sentence 
                    _sample = vocabulary.get_sentence(sample_idxs)
                    #
                    current_num_words = len(sample_idxs)
                    current_word_idxs = np.zeros(config.max_output_length,
                                         dtype = np.int32)
                    current_word_idxs[:current_num_words] = np.array(sample_idxs)
                    current_masks = np.zeros(config.max_output_length)
                    current_masks[:current_num_words] = 1.0
                    maskrl.append(current_masks)
                    sample_feed.append(current_word_idxs)                
                    #
                    ref[l] = vocabulary.get_sentence(dst[l])
                    hypo_greedy[l] = _greedy
                    hypo_sample[l] = _sample
                    #
                    accs_g.append(self._acc(dst[l], greedy_idxs, vocabulary))
                    accs_s.append(self._acc(dst[l], sample_idxs, vocabulary))
                    #
                    idx += 1          
                #
                accs_g = self._acc_seq(np.array(accs_g[:train_data.batch_size-fake_cnt]))
                accs_s = self._acc_seq(np.array(accs_s[:train_data.batch_size-fake_cnt]))
                reward_b = np.array(accs_g[:train_data.batch_size-fake_cnt]) # b
                reward_s = np.array(accs_s[:train_data.batch_size-fake_cnt]) # r
                rewards = 0.1 * (reward_s - reward_b) #
                rewards = np.clip(rewards, -0.1, 0.1)
                #
                maskrl = np.array(maskrl)
                sample_feed = np.array(sample_feed)
                #
                feed_dict = {self.sentences1: _pad(a, config.batch_size), self.sequence_length1: _pad(l_a_, config.batch_size),
                             self.sentences2: _pad(b, config.batch_size), self.sequence_length2: _pad(l_b_, config.batch_size),
                             self.sentences3: _pad(c, config.batch_size), self.sequence_length3: _pad(l_c_, config.batch_size),
                             self.samples: _pad(sample_feed, config.batch_size),
                             self.maskrl: _pad(maskrl, config.batch_size),
                             self.rl_masks2: _pad(m_b, config.batch_size),
                             self.rl_masks3: _pad(m_c, config.batch_size),
                             self.rewards: _pad(rewards, config.batch_size)}
                #
                _, summary, global_step, predic, acc, _labels = sess.run([self.opt_sc,
                                                    self.summary,
                                                    self.global_step, self.rl_predictions, self.rl_accuracy, 
                                                                                   self.rl_labels],
                                                    feed_dict = feed_dict)
                #
                if (global_step + 1) % config.save_period == 0:                   
                    self.save(sess, saver)
                #   
                train_writer.add_summary(summary, global_step)
            train_data.reset()
            #
            config.phase = 'train'
            #
        self.save(sess, saver)           
        train_writer.close()
        print("Training RL complete.")

                
        
    def _acc(self, ref, hypo, vocabulary):
        def _search_right_parenthesis(sent, start, vocabulary):
            l = len(sent)
            while start < l:
                if vocabulary.idx2word[sent[start]] != ')':
                    start += 1
            return start
        l1 = len(ref)
        l2 = len(hypo)
        l = min(l1, l2)
        c, i = 0, 0
        while i < l:
            if ref[i] == hypo[i] or vocabulary.idx2word[hypo[i]] in vocabulary.idx2word[ref[i]]:
                c += 1
            elif vocabulary.idx2word[hypo[i]] == 'cre':
                #if vocabulary.idx2word[ref[i + 1]] == '(':
                #    i = _search_right_parenthesis(ref, i, vocabulary)     
                c += 1
            i += 1
        return c * 1. / l
 

    def _acc_v2(self, ref, hypo):
        def _r(s1):
            s2 = []
            for i in s1:
                if i == 2 or i == 3 or i == 4 or i == 251:
                    continue
                s2.append(i)
            return s2
        ref = _r(ref)
        hypo = _r(hypo)
        l1 = len(ref)
        l2 = len(hypo)
        l = min(l1, l2)
        c = 0
        for i in range(l):
            if ref[i] == hypo[i]:
                c += 1
        return c * 1. / l

    
    def _acc_seq(self, arr_acc):
        y_true = np.ones_like(arr_acc)
        y_pred = arr_acc == y_true
        val = np.mean(y_pred)
        rst = [val] * len(arr_acc)
        return rst
    
    
    def _read_pred_b(self, config):
        filetemp = os.path.join(config.data_dir, "syn_test_1att.hypo_b.d")
        b_hypo = pickle.load(open(filetemp, "rb"))
        b_idx = b_hypo.values()[:-1] # indices
        #        
        word_idxs1 = []
        masks1 = []
        len1 = []
        for sent in tqdm(b_idx):
            current_word_idxs_ = sent
            current_num_words = len(current_word_idxs_)
            #
            len1.append(len(current_word_idxs_))
            print('len(current_word_idxs_)', len(current_word_idxs_), current_word_idxs_)

            current_word_idxs = np.zeros(config.max_output_length,
                                         dtype = np.int32)
            current_masks = np.zeros(config.max_output_length)
            current_word_idxs[:current_num_words] = np.array(current_word_idxs_)
            current_masks[:current_num_words] = 1.0
            word_idxs1.append(current_word_idxs)
            masks1.append(current_masks)
        word_idxs1 = np.array(word_idxs1)
        masks1 = np.array(masks1)    
        len1 = np.array(len1)
        return word_idxs1, masks1, len1
        
        
    def eval(self, sess, eval_data, vocabulary):
        def get_copy_word(sent, vocab2_size): # 
            for i, w in enumerate(sent):
                if w > vocab2_size - 1:
                    return w
                elif w == vocabulary.word2idx["o'clock"]:
                    return sent[i - 1]
            print("copied word not found!!!")
            return vocab2_size - 1
        def get_copy_word2(sent3, hypo_sent4, vocab2_size, vocabulary): # 
            rst = []
            for w in hypo_sent4:
                if w == vocabulary.word2idx["copy"]:
                    rst.append(get_copy_word(sent3, vocab2_size))
                else:
                    rst.append(w)
            return rst
        
        print("Evaluating the model ...")
        config = self.config
        #
        ###b_pre, b_pre_m, b_pre_l = self._read_pred_b(config)
        #
        ref, hypo = {}, {}
        accs, accs2 = [], []
        idx = 0
        #
        for k in tqdm(list(range(eval_data.num_batches)), desc='batch'):
            batch = eval_data.next_batch()
            (a, b, c), (m_a, m_b, m_c), (l_a_, l_b_, l_c_), dst, m_dst, l_dst_ = batch
            #
            batch = (a, b, c), (m_a, m_b, m_c), (l_a_, l_b_, l_c_), dst, m_dst, l_dst_
            decode_data = self.beam_search(sess, batch, vocabulary) 
            
            fake_cnt = 0 if k<eval_data.num_batches-1 \
                         else eval_data.fake_count
            for l in range(eval_data.batch_size-fake_cnt):
                word_idxs = decode_data[l][0].sentence 
                word_idxs = get_copy_word2(c[l], word_idxs, config.vocab2_size, vocabulary)
                score = decode_data[l][0].score 
                forms = vocabulary.get_sentence(word_idxs)
                #
                ref[idx] = vocabulary.get_sentence(dst[l])
                hypo[idx] = forms
                acc = self._acc(dst[l], word_idxs, vocabulary)
                accs.append(acc)
                #acc2 = self._acc_v2(dst[l], word_idxs)
                #accs2.append(acc2)
                idx += 1
        #
        filetemp = os.path.join(config.save_dir, config.eval_result_file)
        pickle.dump(ref, open(filetemp + '.ref.d', 'wb'))
        pickle.dump(hypo, open(filetemp + '.hypo.d', 'wb'))
        results = pd.DataFrame({'idx':hypo.keys(), 
                                'dst':hypo.values()})
        results.to_csv(filetemp + '.hypo.csv')
        #
        print('Acc:', np.mean(accs))
        #
        y_true = np.ones_like(accs)
        y_pred = accs == y_true
        print('Seq Acc', np.mean(y_pred))
        print("Evaluation complete.")

    
    