# ======================================================================================
# 2019
# Project: Context-Dependent Semantic Parsing
# Author: Charles Chen
# Email: lc971015@ohio.edu
# Paper: Context-Dependent Semantic Parsing over Temporally Structured Data, NAACL 2019.
# ======================================================================================

class Config(object):
    def __init__(self):
        # model architecture
        self.max_input_length = 20 
        self.max_output_length = 45 
        self.dim_embedding = 64
        self.num_lstm_units = 64
        self.num_initalize_layers = 2    
        self.dim_initalize_layer = 64
        self.num_attend_layers = 2       
        self.dim_attend_layer = 128 
        self.num_decode_layers = 2       
        self.dim_decode_layer = 64

        # weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # optimization
        self.num_epochs = 6250
        self.batch_size = 32  
        self.optimizer = 'Adam' 
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 4.0 #5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        self.save_period = 100
        self.root_dir = '/home/'
        self.save_dir = self.root_dir + 'log/'
        self.summary_dir = self.root_dir + 'log/'
        self.data_dir = self.root_dir + 'data/'
        
        #
        case = 'synthetic'
        fold = '4'
        self.data_file = case + '.txt'
        
        self.train_dir = self.data_dir + 'train/'
        self.temp_train_file = case + '_' + fold + '_temp_train.npy'
        self.train_file = case + '_train.npy'
        self.vocab1_file = self.train_dir + 'vocab1.txt'
        self.vocab1_size = 738 
        self.vocab2_size = 385 
        
        self.eval_dir = self.data_dir + 'eval/'
        self.temp_eval_file = case + '_' + fold + '_temp_eval.npy'
        self.eval_file = case + '_eval.npy'
        self.eval_result_file = case + '_eval_rst.txt'

        self.test_dir = self.data_dir + 'test/'
        self.temp_test_file = case + '_' + fold + '_temp_test.npy'
        self.test_file = case + '_test.npy'
        self.test_result_file = case + '_test_rst.txt'
        