# ======================================================================================
# 2019
# Project: Context-Dependent Semantic Parsing
# Author: Charles Chen
# Email: lc971015@ohio.edu
# Paper: Context-Dependent Semantic Parsing over Temporally Structured Data, NAACL 2019.
# ======================================================================================

#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import Model
from dataset import prepare_data

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv):
    config = Config()
    config.phase = FLAGS.phase
    config.beam_size = FLAGS.beam_size
    with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            train_data, vocabulary1 = prepare_data(config)
            model = Model(config, vocabulary1)
            saver = tf.train.Saver(max_to_keep = 1000)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, saver, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.train(sess, saver, train_data, vocabulary1)
        elif FLAGS.phase == 'eval':
            # evaluation phase
            eval_data, vocabulary = prepare_data(config)
            model = Model(config, vocabulary)
            saver = tf.train.Saver(max_to_keep = 1000)
            model.load(sess, saver, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, eval_data, vocabulary)
        else:
            # testing phase
            test_data, vocabulary = prepare_data(config)
            model = Model(config, vocabulary)
            saver = tf.train.Saver(max_to_keep = 1000)
            model.load(sess, saver, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, test_data, vocabulary)

if __name__ == '__main__':
    tf.app.run()



