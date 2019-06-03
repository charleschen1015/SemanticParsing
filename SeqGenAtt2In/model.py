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
                                       units = config.num_lstm_units * 2,
                                       activation = None,
                                       name = 'fc_a')
            output = self.nn.dense(encoding2,
                                       units = config.num_lstm_units * 2,
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

        contexts = tf.concat(outputs, 2) #
        print("contexts", contexts)

        print("output_states", output_states)
        output_state_fw, output_state_bw = output_states
        t1 = tf.concat((output_state_fw[0], output_state_bw[0]),  1)
        t2 = tf.concat((output_state_fw[1], output_state_bw[1]),  1)
        new_encode_state = t1, t2
        print("encode_state", new_encode_state)
               
        print("RNN1 built.")
        return contexts, new_encode_state
       
    
    def build_encoder(self):
        config = self.config

        self.sentences3 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size, config.max_input_length])
        self.sequence_length3 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size])
        #
        self.contexts3, self.encode_state1 = self.build_rnn1(self.sentences3, self.sequence_length3, self.encode_state1)
        print('Encoder built..')
        
              
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
    
    
    def decode_v2(self, output):
        config = self.config
        v1 = output # generate v1: [batch_size, long]

        # generate
        logits1 = self.nn.dense(v1,
                                units = config.vocab2_size, # logical forms tokens
                                activation = None,
                                name = 'generate') # logits1: [batch_size, vocab2_size]
        return logits1
    
        
    def get_l(self, sent4, sentences2, sentences3):
        # sent4: [batch_size] generate
        # sentences2: [batch_size, max_output_length] cre coref
        # sentences3: [batch_size, max_input_length]  copy 
        config = self.config
        l = []
        for idx, w in enumerate(sent4):
            l1 = tf.zeros([config.vocab2_size], tf.float32) # 
            l2 = tf.zeros([config.max_output_length], tf.float32) #  coref
            l3 = tf.zeros([config.max_input_length], tf.float32) #   copy
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
    
    
    
    def get_l_v2(self, sent4):
        config = self.config
                
        masked = tf.less(sent4, config.vocab2_size)
        oov = tf.zeros_like(sent4) + config.vocab2_size - 1
        t1 = tf.where(masked, sent4, oov)
        t1 = tf.one_hot(t1, config.vocab2_size) # [batch_size, vocab2_size]
        l1 = t1
        
        l1 = tf.argmax(l1, axis = 1)
        return l1
    
    
    
    def build_rnn3(self):
        config = self.config

        if self.is_train:
            sentences4 = tf.placeholder(dtype = tf.int32, shape = [config.batch_size, config.max_output_length])
            #
            masks4 = tf.placeholder(dtype = tf.float32, shape = [config.batch_size, config.max_output_length])
            #
        else:
            dim_ctx = config.num_lstm_units * 2
                    
            last_memory = tf.placeholder(dtype = tf.float32,
                                         shape = [config.batch_size, config.num_lstm_units * 2])
            last_output = tf.placeholder(dtype = tf.float32,
                                         shape = [config.batch_size, config.num_lstm_units * 2])
            last_word = tf.placeholder(dtype = tf.int64,
                                         shape = [config.batch_size])

        with tf.variable_scope("word_embedding", reuse = tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocab1_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = self.is_train)
                
        lstm = tf.contrib.rnn.LSTMCell(
            config.num_lstm_units * 2,
            initializer = self.nn.fc_kernel_initializer, state_is_tuple = True)

        if self.is_train:
            lstm = tf.contrib.rnn.DropoutWrapper(
                lstm,
                input_keep_prob = 1.0 - config.lstm_drop_rate,
                output_keep_prob = 1.0 - config.lstm_drop_rate,
                state_keep_prob = 1.0 - config.lstm_drop_rate)
            print('rnn3 use dropout')


        with tf.variable_scope("initialize3_1"):
            initial_state1 = self.initialize(self.encode_state1)
            initial_memory, initial_output = initial_state1
            initial_state = tf.contrib.rnn.LSTMStateTuple(initial_memory, initial_output)
            print("initial_state", initial_state)        
        
        
        predictions = []
        if self.is_train:
            _labels, cross_entropies = [], []
            predictions_correct = []
            num_steps = config.max_output_length
            last_output = initial_output
            last_memory = initial_memory
            last_word = tf.zeros([config.batch_size], tf.int32)
        else:
            num_steps = 1
        last_state = last_memory, last_output
        
        
                
        for idx in range(num_steps):                                            
            #
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)

            with tf.variable_scope("rnn3"):
                current_input = word_embed #tf.concat([word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             word_embed],
                                             axis = 1)
                logits1 = self.decode_v2(output) 
                probs = tf.nn.softmax(logits1) #logits  batch_size * vocab2_size
                #
                prediction = tf.argmax(logits1, 1) #_prediction batch_size                               
                #
                predictions.append(prediction)

                
            if self.is_train:
                labels1 = self.get_l_v2(sentences4[:, idx]) # [batch_size, long]
                _labels.append(labels1)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = labels1, #sentences4[:, idx],
                    logits = logits1)
                masked_cross_entropy = cross_entropy * masks4[:, idx]
                cross_entropies.append(masked_cross_entropy)
               
            
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
        if self.is_train:
            self._labels = tf.stack(_labels, axis = 1)
            cross_entropies = tf.stack(cross_entropies, axis = 1)
            cross_entropy_loss = tf.reduce_sum(cross_entropies) \
                                 / tf.reduce_sum(masks4)
                                         
                    
            #
            reg_loss = tf.losses.get_regularization_loss()
            total_loss = cross_entropy_loss + reg_loss 
            
            predictions_correct = tf.stack(predictions_correct, axis = 1) # [batch_size, max_caption_length]
            accuracy = tf.reduce_sum(tf.cast(predictions_correct, tf.float32)) \
                       / tf.reduce_sum(masks4)
        
        self.predictions = tf.stack(predictions, axis = 1)
        if self.is_train:
            self.sentences4 = sentences4
            self.masks4 = masks4
            self.total_loss = total_loss
            self.cross_entropy_loss = cross_entropy_loss
                        
            self.reg_loss = reg_loss
            self.accuracy = accuracy
            self.perplexity = tf.exp(cross_entropy_loss)
        else:
            self.initial_memory = initial_memory
            self.initial_output = initial_output

            self.last_memory = last_memory
            self.last_output = last_output
            self.last_word = last_word
            self.memory = memory
            self.output = output
            self.probs = probs    
        print("RNN3 built.")
            


    def optimizer(self):
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
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

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op
        print('Optimizor built..')


    def summary(self):
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)

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
                #print(_, a[0], b[0], c[0], m_a[0], m_b[0], m_c[0],  l_a_[0], l_b_[0], l_c_[0], dst[0], m_dst[0], l_dst_[0])
                feed_dict = {self.sentences3: c, self.sequence_length3: l_c_,
                             self.sentences4: dst, self.masks4: m_dst}
                _, summary, global_step, predic, acc, _labels = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step, self.predictions, self.accuracy, 
                                                                                   self._labels],
                                                    feed_dict = feed_dict)
    
                #
                if (global_step + 1) % config.save_period == 0:
                    print('l', _labels[:5, :])
                    print('dst', dst[:5, :])
                    print('sent3', c[:5, :])
                    print(vocabulary.get_sentence_batch(predic) )  
                    
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

        
        
    def _acc(self, ref, hypo, vocabulary):
        l1 = len(ref)
        l2 = len(hypo)
        l = min(l1, l2)
        c, i = 0, 0
        while i < l:
            if ref[i] == hypo[i] or vocabulary.idx2word[hypo[i]] in vocabulary.idx2word[ref[i]]:
                c += 1
            elif vocabulary.idx2word[hypo[i]] == 'cre':    
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


        
    def eval(self, sess, eval_data, vocabulary):        
        print("Evaluating the model ...")
        config = self.config
        #
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
                score = decode_data[l][0].score 
                forms = vocabulary.get_sentence(word_idxs)
                #
                ref[idx] = vocabulary.get_sentence(dst[l])
                hypo[idx] = forms
                acc = self._acc(dst[l], word_idxs, vocabulary)
                accs.append(acc)
                #acc2 = self._acc_v2(dst[l], word_idxs)
                #accs2.append(acc2)
                print(idx, vocabulary.get_sentence(c[l]), ref[idx], hypo[idx], acc)
                idx += 1
        #
        filetemp = os.path.join(config.eval_dir, config.eval_result_file)
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
        #y_pred2 = accs2 == y_true
        print('Seq Acc', np.mean(y_pred))
        print("Evaluation complete.")


